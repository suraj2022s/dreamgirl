#!/usr/bin/env python3
"""
Telegram UPI Bot — Join gate, dynamic plans (/prices), fixed roster (/fixedmodels),
UPI deeplink as text, QR code, and manual proof (Unique Transaction Reference number + screenshot) to admin(s).

Highlights
- Join-gate (optional) requiring users to be members of a channel before using the bot
- /prices: dynamic plans list from models.json (auto-sorted by price)
- /fixedmodels: fixed roster with photos, global fixed price, pagination, QR + proof buttons
- Admin CRUD for fixed roster: /fixedadd, /fixeddel, /setfixedprice, /fixedlist
- Payment proof flow: collect UTR (Unique Transaction Reference number) + optional screenshot
- Recording payments -> payments.jsonl, /total YYYY-MM-DD, optional daily summary (DAILY_SUM_HHMM)
- Multi-admin support via ADMIN_CHAT_IDS (and VERIFIER_CHAT_IDS)
- Polling mode (local) and Webhook mode (Render/any web host)
- Self-test: run `python bot.py selftest`
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import string
import threading
import urllib.parse
from datetime import datetime, time as dtime, timezone, timedelta
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from zoneinfo import ZoneInfo

# Third-party runtime deps
try:
    import qrcode  # pillow required
except Exception:
    qrcode = None

# Telegram (optional import guard for selftest environments)
try:
    from telegram import (
        Update,
        InlineKeyboardMarkup,
        InlineKeyboardButton,
        ChatMember,
    )
    from telegram.constants import ChatMemberStatus, ParseMode
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    TELEGRAM_AVAILABLE = True
except ModuleNotFoundError:
    TELEGRAM_AVAILABLE = False

# ----------------------- ENV / CONFIG ----------------------- #

def _getenv(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else default

# Robust base dir resolution: works even if __file__ is undefined
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive/sandboxed
    BASE_DIR = os.getcwd()

BOT_TOKEN = _getenv("BOT_TOKEN")
CHANNEL_USERNAME = _getenv("CHANNEL_USERNAME", "@YourChannel")  # e.g. @public_channel or -100.. for private
UPI_VPA = _getenv("UPI_VPA", "merchant@upi")
UPI_NAME = _getenv("UPI_NAME", "Your Store")
MODELS_DB_RAW = _getenv("MODELS_DB", "models.json")
FIXED_DB_RAW = _getenv("FIXED_DB", "fixed_models.json")
PAYMENTS_DB_RAW = _getenv("PAYMENTS_DB", "payments.jsonl")
REQUIRE_JOIN = _getenv("REQUIRE_JOIN", "0") in {"1", "true", "True", "yes", "YES"}
TIMEZONE = _getenv("TIMEZONE", "Asia/Kolkata")
DAILY_SUM_HHMM = _getenv("DAILY_SUM_HHMM", "")  # e.g. 23:55

# Webhook support (Render/web services)
USE_WEBHOOK = _getenv("USE_WEBHOOK", "0") in {"1", "true", "True", "yes", "YES"}
WEBHOOK_URL = _getenv("WEBHOOK_URL", "")  # e.g. https://your-service.onrender.com
PORT = int(_getenv("PORT", "8000"))
SECRET_TOKEN = _getenv("WEBHOOK_SECRET", "")

# Resolve paths relative to script folder if not absolute
MODELS_DB = MODELS_DB_RAW if os.path.isabs(MODELS_DB_RAW) else os.path.join(BASE_DIR, MODELS_DB_RAW)
FIXED_DB = FIXED_DB_RAW if os.path.isabs(FIXED_DB_RAW) else os.path.join(BASE_DIR, FIXED_DB_RAW)
PAYMENTS_DB = PAYMENTS_DB_RAW if os.path.isabs(PAYMENTS_DB_RAW) else os.path.join(BASE_DIR, PAYMENTS_DB_RAW)

ADMIN_CHAT_IDS_RAW = _getenv("ADMIN_CHAT_IDS", "")
VERIFIER_CHAT_IDS_RAW = _getenv("VERIFIER_CHAT_IDS", "")


def _parse_ids(raw: str) -> set[int]:
    ids: set[int] = set()
    for part in re.split(r"[,\s]+", raw.strip()):
        if not part:
            continue
        try:
            ids.add(int(part))
        except Exception:
            pass
    return ids

ADMIN_CHAT_IDS: set[int] = _parse_ids(ADMIN_CHAT_IDS_RAW)
VERIFIER_CHAT_IDS: set[int] = _parse_ids(VERIFIER_CHAT_IDS_RAW)

# Back-compat single ID envs
try:
    _single_admin = int(_getenv("ADMIN_CHAT_ID", "0") or "0")
except ValueError:
    _single_admin = 0
if _single_admin:
    ADMIN_CHAT_IDS.add(_single_admin)

try:
    _single_verifier = int(_getenv("VERIFIER_CHAT_ID", "0") or "0")
except ValueError:
    _single_verifier = 0
if _single_verifier:
    VERIFIER_CHAT_IDS.add(_single_verifier)

VERIFIER_HANDLE = _getenv("VERIFIER_HANDLE", "")

WELCOME_BANNER = """
<b>Welcome!</b>
Use the menu to view plans and pay via UPI.

<b>Steps</b>
1) (If required) Join our channel
2) Pick a plan or a model
3) Pay via UPI (deeplink or QR)
4) Tap <i>I've Paid</i>, send Unique Transaction Reference number + screenshot
5) We'll confirm and schedule
"""

# ----------------------- TIME/ZONE ----------------------- #

def _get_zone(tzname: str):
    try:
        return ZoneInfo(tzname)
    except Exception:
        # Fallback for systems without tzdata
        if tzname in ("Asia/Kolkata", "Asia/Calcutta"):
            return timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
        return timezone.utc

ZONE = _get_zone(TIMEZONE)

def now_local() -> datetime:
    return datetime.now(ZONE)

# ----------------------- STORAGE LOADERS ----------------------- #

MODELS: list[dict] = []  # dynamic plans

# Default models (your catalog)
DEFAULT_MODELS = [
    {"id": "plan1", "name": "Pic In Dress", "price_inr": 200, "sku": "PIC1", "desc": "1 photo in dress"},
    {"id": "plan2", "name": "Nude Pic", "price_inr": 500, "sku": "NUDE1", "desc": "1 nude photo"},
    {"id": "plan3", "name": "Audio Call", "price_inr": 500, "sku": "CALL8", "desc": "8-minute audio call"},
    {"id": "plan4", "name": "Boobs Show", "price_inr": 500, "sku": "BOOBS5", "desc": "5-minute boobs show"},
    {"id": "plan5", "name": "Full Show", "price_inr": 1500, "sku": "FULL12", "desc": "12-minute full show"},
    {"id": "plan6", "name": "Sex/Dirty Chat", "price_inr": 500, "sku": "CHAT8", "desc": "8-minute dirty chat"},
    {"id": "plan7", "name": "Full Show with Face", "price_inr": 2000, "sku": "FULLFACE15", "desc": "15-minute full show with face"},
    {"id": "plan8", "name": "Fingering Show", "price_inr": 1500, "sku": "FINGER10", "desc": "10-minute fingering show"},
    {"id": "plan9", "name": "Abuse Show", "price_inr": 1500, "sku": "ABUSE12", "desc": "12-minute abuse show"},
    {"id": "plan10", "name": "Role Play", "price_inr": 1500, "sku": "ROLE12", "desc": "12-minute role play"},
    {"id": "plan11", "name": "Couple Show", "price_inr": 3500, "sku": "COUPLE30", "desc": "30-minute couple show"},
]

DEFAULT_FIXED = {
    "fixed_price": {"amount": 2000, "duration": "15 min"},
    "models": [
        {"id": "m1", "name": "Meera", "photo": "https://picsum.photos/seed/meera/800/1000", "desc": "Fixed-rate session"}
    ],
}


def ensure_default_files() -> None:
    """Create default JSON files if missing."""
    if not os.path.exists(MODELS_DB):
        try:
            with open(MODELS_DB, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_MODELS, f, ensure_ascii=False, indent=2)
            logging.info(f"Created default models file at {MODELS_DB}")
        except Exception as e:
            logging.error(f"Failed to create {MODELS_DB}: {e}")
    if not os.path.exists(FIXED_DB):
        try:
            with open(FIXED_DB, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_FIXED, f, ensure_ascii=False, indent=2)
            logging.info(f"Created default fixed models file at {FIXED_DB}")
        except Exception as e:
            logging.error(f"Failed to create {FIXED_DB}: {e}")


def _price(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def load_models() -> None:
    """Load plans from models.json; keep a sorted list by price asc then name."""
    global MODELS
    try:
        with open(MODELS_DB, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            MODELS = sorted(
                data,
                key=lambda m: (_price(m.get("price_inr", 0)), str(m.get("name", "")).lower()),
            )
        else:
            logging.warning("models.json is not a JSON array; using empty list")
            MODELS = []
    except FileNotFoundError:
        logging.warning("models.json not found — using empty list")
        MODELS = []
    except Exception as e:
        logging.exception(f"Failed to load models.json: {e}")
        MODELS = []


def load_fixed() -> dict:
    try:
        with open(FIXED_DB, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("fixed_models.json root must be an object")
        data.setdefault("fixed_price", {"amount": 2000, "duration": "15 min"})
        data.setdefault("models", [])
        return data
    except FileNotFoundError:
        logging.warning("fixed_models.json not found — using defaults")
        return {"fixed_price": {"amount": 2000, "duration": "15 min"}, "models": []}
    except Exception as e:
        logging.exception(f"Failed to load fixed_models.json: {e}")
        return {"fixed_price": {"amount": 2000, "duration": "15 min"}, "models": []}


def save_fixed(data: dict) -> None:
    try:
        with open(FIXED_DB, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception(f"Failed to save fixed_models.json: {e}")

# ----------------------- HELPERS ----------------------- #

def _rand_tag(n: int = 4) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def make_upi_link(vpa: str, name: str, amount_inr: float, note: str) -> str:
    params = {
        "pa": vpa,
        "pn": name,
        "am": f"{float(amount_inr):.2f}",
        "cu": "INR",
        "tn": note,
    }
    return "upi://pay?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def make_qr_png(data: str) -> bytes:
    if qrcode is None:
        raise RuntimeError("qrcode library not installed. Run: pip install qrcode[pil]")
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


async def is_user_in_channel(app: 'Application', user_id: int) -> bool:
    if not REQUIRE_JOIN:
        return True
    try:
        member: 'ChatMember' = await app.bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=user_id)
        return member.status in {ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER}
    except Exception as e:
        logging.warning(f"get_chat_member failed: {e}")
        return False


def is_admin(update: 'Update') -> bool:
    uid = update.effective_user.id if update.effective_user else 0
    return uid in ADMIN_CHAT_IDS


def build_fixed_caption(model: dict, amt: int, dur: str, upi_url: str) -> str:
    return (
        f"<b>{model['name']}</b> — ₹{amt} / {dur}\n"
        f"{model.get('desc','')}\n\n"
        f"UPI: <code>{upi_url}</code>"
    )

# ----------------------- KEYBOARDS ----------------------- #

def price_keyboard() -> 'InlineKeyboardMarkup':
    rows = []
    for idx, m in enumerate(MODELS, start=1):
        rows.append([InlineKeyboardButton(f"{idx}. {m['name']} — ₹{m['price_inr']}", callback_data=f"detail:{m['id']}")])
    if not rows:
        rows = [[InlineKeyboardButton("(no plans configured)", callback_data="noop")]]
    rows.append([InlineKeyboardButton("📖 Fixed Models", callback_data="goto:fixed")])
    return InlineKeyboardMarkup(rows)


def model_action_keyboard(mid: str) -> 'InlineKeyboardMarkup':
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔲 Show QR", callback_data=f"qr:{mid}")],
        [InlineKeyboardButton("✅ I've Paid", callback_data=f"proof:{mid}")],
        [InlineKeyboardButton("⬅️ Back to Price List", callback_data="back:prices")],
    ])

# Pagination and cards for fixed models list
PAGE_SIZE = 3


def fixed_nav_keyboard(page: int, total_pages: int) -> 'InlineKeyboardMarkup':
    buttons = []
    row = []
    if page > 0:
        row.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"fixedpage:{page-1}"))
    if page < total_pages - 1:
        row.append(InlineKeyboardButton("Next ➡️", callback_data=f"fixedpage:{page+1}"))
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("⬅️ Back to Price List", callback_data="back:prices")])
    return InlineKeyboardMarkup(buttons)


def fixed_card_keyboard(mid: str) -> 'InlineKeyboardMarkup':
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("ℹ️ Details", callback_data=f"fdetail:{mid}")],
        [InlineKeyboardButton("🔲 Show QR", callback_data=f"fqr:{mid}"), InlineKeyboardButton("✅ I've Paid", callback_data=f"fproof:{mid}")],
    ])


def fixed_admin_delete_keyboard() -> 'InlineKeyboardMarkup':
    db = load_fixed()
    rows = []
    for m in db.get("models", []):
        name = m.get("name", "(unnamed)")
        mid = m.get("id", "")
        rows.append([InlineKeyboardButton(f"🗑️ {name}", callback_data=f"fdel:{mid}")])
    if not rows:
        rows = [[InlineKeyboardButton("(no models configured)", callback_data="noop")]]
    return InlineKeyboardMarkup(rows)

# ----------------------- COMMANDS ----------------------- #

async def start(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    uid = update.effective_user.id if update.effective_user else 0
    if not await is_user_in_channel(context.application, uid):
        await update.effective_message.reply_html(
            f"Please join our channel first: <a href='https://t.me/{CHANNEL_USERNAME.lstrip('@')}'>open</a>")
        return
    await send_welcome_and_prices(update, context)


async def help_cmd(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    extra = ""
    if is_admin(update):
        extra = (
            "\n<b>Admin</b>\n"
            "/fixedadd — add a fixed model (photo→name→desc)\n"
            "/fixeddel — delete a fixed model\n"
            "/setfixedprice &lt;amount&gt; &lt;duration text&gt;\n"
            "/fixedlist — show models count and price\n"
            "/total [YYYY-MM-DD] — sum for date\n"
        )
    await update.effective_message.reply_html(
        (
            "<b>Commands</b>\n"
            "/start — show welcome\n"
            "/prices — plans list\n"
            "/fixedmodels — fixed-price roster\n"
            "/reload — reload files & show counts\n"
        ) + extra
    )


async def send_welcome_and_prices(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id=chat_id, text=WELCOME_BANNER, parse_mode=ParseMode.HTML)
    load_models()
    lines = ["<b>Price List</b>"]
    for idx, m in enumerate(MODELS, start=1):
        lines.append(f"{idx}) <b>{m['name']}</b> — ₹{m['price_inr']}\n<i>{m['desc']}</i>")
    await context.bot.send_message(
        chat_id=chat_id,
        text="\n\n".join(lines) if len(lines) > 1 else "No plans configured.",
        parse_mode=ParseMode.HTML,
        reply_markup=price_keyboard(),
    )


async def prices_cmd(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    uid = update.effective_user.id if update.effective_user else 0
    if not await is_user_in_channel(context.application, uid):
        await update.effective_message.reply_html(
            f"Please join our channel first: <a href='https://t.me/{CHANNEL_USERNAME.lstrip('@')}'>open</a>")
        return
    await send_welcome_and_prices(update, context)


async def fixedmodels_cmd(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    uid = update.effective_user.id if update.effective_user else 0
    if not await is_user_in_channel(context.application, uid):
        await update.effective_message.reply_html(
            f"Please join our channel first: <a href='https://t.me/{CHANNEL_USERNAME.lstrip('@')}'>open</a>")
        return
    await render_fixed_page(update, context, page=0)


async def reload_cmd(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    load_models()
    fdata = load_fixed()
    await update.effective_message.reply_text(
        (
            "Reloaded files:\n"
            f"models.json → {MODELS_DB} (plans: {len(MODELS)})\n"
            f"fixed_models.json → {FIXED_DB} (models: {len(fdata['models'])}, price: ₹{fdata['fixed_price']['amount']}/{fdata['fixed_price']['duration']})"
        )
    )

# ----------------------- ADMIN: Fixed Models CRUD ----------------------- #

async def fixed_add_cmd(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    if not is_admin(update):
        await update.effective_message.reply_text("Admins only.")
        return
    context.user_data["add_fixed"] = {"step": "photo"}
    await update.effective_message.reply_text("Send the model photo now, or type 'cancel' to abort.")


async def fixed_del_cmd(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    if not is_admin(update):
        await update.effective_message.reply_text("Admins only.")
        return
    await update.effective_message.reply_html("Tap a model to delete:", reply_markup=fixed_admin_delete_keyboard())


async def set_fixed_price_cmd(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    """Set the global fixed price.
    Formats:
      /setfixedprice 2000 15 min
      /setfixedprice 2000
      /setfixedprice ₹2,499 20 minutes
    If duration omitted, keep existing duration.
    """
    if not is_admin(update):
        await update.effective_message.reply_text("Admins only.")
        return

    text = (update.effective_message.text or "").strip()
    rest = text[len("/setfixedprice"):].strip()
    if not rest:
        await update.effective_message.reply_text(
            "Usage: /setfixedprice <amount> <duration text>\n"
            "Examples:\n"
            "• /setfixedprice 2000 15 min\n"
            "• /setfixedprice 2499 20 minutes\n"
            "• /setfixedprice 2000  (keeps current duration)"
        )
        return

    rest_clean = rest.replace("₹", "").replace(",", " ")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", rest_clean)
    if not m:
        await update.effective_message.reply_text("Couldn't find the amount. Example: /setfixedprice 2000 15 min")
        return
    amount_str = m.group(1)
    try:
        amount = int(float(amount_str))
    except Exception:
        await update.effective_message.reply_text("Amount must be a number, e.g. 2000")
        return

    tail = rest_clean[m.end():].strip()

    data = load_fixed()
    current_duration = data.get("fixed_price", {}).get("duration", "15 min")
    duration = tail if tail else current_duration

    data["fixed_price"] = {"amount": amount, "duration": duration}
    save_fixed(data)

    await update.effective_message.reply_text(f"Updated fixed price to ₹{amount} / {duration}")


async def fixed_list_cmd(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    if not is_admin(update):
        await update.effective_message.reply_text("Admins only.")
        return
    data = load_fixed()
    lines = [
        f"Price: ₹{data['fixed_price']['amount']} / {data['fixed_price']['duration']}",
        f"Models: {len(data.get('models', []))}",
    ]
    for m in data.get("models", []):
        lines.append(f"- {m['name']} ({m['id']})")
    await update.effective_message.reply_text("\n".join(lines))

# ----------------------- FIXED LIST RENDER ----------------------- #

async def render_fixed_page(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE', page: int) -> None:
    data = load_fixed()
    models = data.get("models", [])
    amount = data["fixed_price"]["amount"]
    duration = data["fixed_price"]["duration"]

    total = len(models)
    if total == 0:
        await update.effective_message.reply_html("No fixed models available.")
        return

    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, total_pages - 1))

    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    page_models = models[start:end]

    header = f"<b>Fixed Models</b> — ₹{amount} / {duration}\nPage {page+1}/{total_pages}"
    await update.effective_message.reply_html(header, reply_markup=fixed_nav_keyboard(page, total_pages))

    for m in page_models:
        upi_url = make_upi_link(UPI_VPA, UPI_NAME, amount, m["name"])  # same price for all
        cap = build_fixed_caption(m, amount, duration, upi_url)
        photo = m.get("photo")
        if photo:
            try:
                await update.effective_message.reply_photo(
                    photo=photo,
                    caption=cap,
                    parse_mode=ParseMode.HTML,
                    reply_markup=fixed_card_keyboard(m["id"]),
                )
                continue
            except Exception as e:
                logging.warning(f"Failed to send model photo in list: {e}")
        # fallback to text card
        await update.effective_message.reply_html(cap, reply_markup=fixed_card_keyboard(m["id"]))

# ----------------------- CALLBACKS ----------------------- #

async def on_callback(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    q = update.callback_query
    data = q.data or ""

    # navigation
    if data == "goto:fixed":
        await render_fixed_page(update, context, page=0)
        await q.answer()
        return

    if data == "back:prices":
        await send_welcome_and_prices(update, context)
        await q.answer()
        return

    if data == "back:fixed":
        await render_fixed_page(update, context, page=0)
        await q.answer()
        return

    if data.startswith("fixedpage:"):
        try:
            page = int(data.split(":", 1)[1])
        except Exception:
            page = 0
        await render_fixed_page(update, context, page=page)
        await q.answer()
        return

    if data == "noop":
        await q.answer()
        return

    # Admin delete fixed model
    if data.startswith("fdel:"):
        if not is_admin(update):
            await q.answer("Admins only", show_alert=True)
            return
        mid = data.split(":", 1)[1]
        db = load_fixed()
        before = len(db.get("models", []))
        db["models"] = [m for m in db.get("models", []) if m.get("id") != mid]
        after = len(db.get("models", []))
        save_fixed(db)
        await q.answer("Deleted" if after < before else "Not found")
        await q.message.reply_text("Updated.")
        return

    # Dynamic plans
    if data.startswith("detail:"):
        mid = data.split(":", 1)[1]
        load_models()
        model = next((m for m in MODELS if m["id"] == mid), None)
        if not model:
            await q.answer("Model not found", show_alert=True)
            return
        upi_url = make_upi_link(UPI_VPA, UPI_NAME, model['price_inr'], f"{model['name']} {model.get('sku','')}")
        text = (
            f"<b>{model['name']}</b> — ₹{model['price_inr']}\n{model['desc']}\n\n"
            f"SKU: <code>{model.get('sku','')}</code>\n"
            f"UPI: <code>{upi_url}</code>"
        )
        await q.message.reply_html(text, reply_markup=model_action_keyboard(mid))
        await q.answer()
        return

    if data.startswith("qr:"):
        mid = data.split(":", 1)[1]
        load_models()
        model = next((m for m in MODELS if m["id"] == mid), None)
        if not model:
            await q.answer("Model not found", show_alert=True)
            return
        upi_url = make_upi_link(UPI_VPA, UPI_NAME, model['price_inr'], f"{model['name']} {model.get('sku','')}")
        png = make_qr_png(upi_url)
        await q.message.reply_photo(photo=png, caption=f"Scan to pay ₹{model['price_inr']}\nVPA: {UPI_VPA}")
        await q.answer("QR ready")
        return

    if data.startswith("proof:"):
        mid = data.split(":", 1)[1]
        load_models()
        model = next((m for m in MODELS if m["id"] == mid), None)
        if not model:
            await q.answer("Model not found", show_alert=True)
            return
        order_id = f"ODR-{datetime.now().strftime('%y%m%d')}-{_rand_tag()}"
        context.user_data["awaiting_proof"] = {
            "type": "plan",
            "model_id": model["id"],
            "model_name": model["name"],
            "price_inr": model["price_inr"],
            "sku": model.get("sku", ""),
            "order_id": order_id,
        }
        await q.message.reply_html(
            f"<b>Payment Proof</b> for <b>{model['name']}</b> (₹{model['price_inr']})\n"
            f"Order: <code>{order_id}</code>\n\n"
            "Send your <b>Unique Transaction Reference number</b> now. Then send a <b>payment screenshot</b> (optional).\n"
            "Type <code>cancel</code> to abort."
        )
        await q.answer()
        return

    # Fixed roster (user-facing detail)
    if data.startswith("fdetail:"):
        mid = data.split(":", 1)[1]
        fdata = load_fixed()
        model = next((m for m in fdata["models"] if m["id"] == mid), None)
        if not model:
            await q.answer("Model not found", show_alert=True)
            return
        amt = fdata["fixed_price"]["amount"]
        dur = fdata["fixed_price"]["duration"]
        upi_url = make_upi_link(UPI_VPA, UPI_NAME, amt, model["name"])
        caption = build_fixed_caption(model, amt, dur, upi_url)
        photo = model.get("photo")
        if photo:
            try:
                await q.message.reply_photo(
                    photo=photo,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    reply_markup=fixed_card_keyboard(mid),
                )
                await q.answer()
                return
            except Exception as e:
                logging.warning(f"Failed to send model photo: {e}")
        await q.message.reply_html(caption, reply_markup=fixed_card_keyboard(mid))
        await q.answer()
        return

    if data.startswith("fqr:"):
        mid = data.split(":", 1)[1]
        fdata = load_fixed()
        model = next((m for m in fdata["models"] if m["id"] == mid), None)
        if not model:
            await q.answer("Model not found", show_alert=True)
            return
        amt = fdata["fixed_price"]["amount"]
        upi_url = make_upi_link(UPI_VPA, UPI_NAME, amt, model["name"])
        png = make_qr_png(upi_url)
        await q.message.reply_photo(photo=png, caption=f"Scan to pay ₹{amt}\nVPA: {UPI_VPA}")
        await q.answer("QR ready")
        return

    if data.startswith("fproof:"):
        mid = data.split(":", 1)[1]
        fdata = load_fixed()
        model = next((m for m in fdata["models"] if m["id"] == mid), None)
        if not model:
            await q.answer("Model not found", show_alert=True)
            return
        amt = fdata["fixed_price"]["amount"]
        order_id = f"ODR-{datetime.now().strftime('%y%m%d')}-{_rand_tag()}"
        context.user_data["awaiting_proof"] = {
            "type": "fixed",
            "model_id": model["id"],
            "model_name": model["name"],
            "price_inr": amt,
            "sku": model.get("id", ""),
            "order_id": order_id,
        }
        await q.message.reply_html(
            f"<b>Payment Proof</b> for <b>{model['name']}</b> (₹{amt})\n"
            f"Order: <code>{order_id}</code>\n\n"
            "Send your <b>Unique Transaction Reference number</b> now. Then send a <b>payment screenshot</b> (optional).\n"
            "Type <code>cancel</code> to abort."
        )
        await q.answer()
        return

# ----------------------- MESSAGE HANDLERS ----------------------- #

async def on_text(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    txt = (update.effective_message.text or "").strip()
    if not txt:
        return

    # ADMIN: add fixed model flow
    add = context.user_data.get("add_fixed")
    if add:
        if txt.lower() == "cancel":
            context.user_data.pop("add_fixed", None)
            await update.effective_message.reply_text("Cancelled.")
            return
        if add.get("step") == "name":
            add["name"] = txt
            add["step"] = "desc"
            context.user_data["add_fixed"] = add
            await update.effective_message.reply_text("Enter a short description (or '-' to skip).")
            return
        if add.get("step") == "desc":
            add["desc"] = "" if txt == "-" else txt
            db = load_fixed()
            new_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
            db.setdefault("models", []).append({
                "id": new_id,
                "name": add.get("name", "Unnamed"),
                "photo": add.get("photo_file_id", ""),
                "desc": add.get("desc", ""),
            })
            save_fixed(db)
            context.user_data.pop("add_fixed", None)
            await update.effective_message.reply_html(
                f"Added <b>{db['models'][-1]['name']}</b> with id <code>{new_id}</code>.\nUse /fixedmodels to view.")
            return

    # payment proof conversation
    pending = context.user_data.get("awaiting_proof")
    if pending:
        if txt.lower() == "cancel":
            context.user_data.pop("awaiting_proof", None)
            context.user_data.pop("awaiting_screenshot", None)
            await update.effective_message.reply_text("Cancelled.")
            return
        if txt.lower() == "skip":
            await _finalize_proof(update, context, with_photo=False)
            return
        pending["utr"] = txt
        context.user_data["awaiting_proof"] = pending
        context.user_data["awaiting_screenshot"] = True
        await update.effective_message.reply_text("✅ Got the Unique Transaction Reference number. Now send a screenshot (or type 'skip').")
        return

    await help_cmd(update, context)


async def on_photo(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    add = context.user_data.get("add_fixed")
    if add and add.get("step") == "photo":
        if not is_admin(update):
            await update.effective_message.reply_text("Admins only.")
            return
        photo = update.message.photo[-1]
        add["photo_file_id"] = photo.file_id
        add["step"] = "name"
        context.user_data["add_fixed"] = add
        await update.effective_message.reply_text("Photo saved. Now enter the model name.")
        return

    if context.user_data.get("awaiting_screenshot") and context.user_data.get("awaiting_proof"):
        photo = update.message.photo[-1]
        context.user_data["screenshot_file_id"] = photo.file_id
        await _finalize_proof(update, context, with_photo=True)
        return

    await help_cmd(update, context)

# ----------------------- PAYMENTS LOGIC ----------------------- #

def record_payment(proof: dict, user_id: int, user_name: str) -> None:
    """Append a JSON line with the payment info for later reporting."""
    try:
        ts = now_local()
        row = {
            "ts": ts.isoformat(),
            "date": ts.strftime("%Y-%m-%d"),
            "amount": int(proof.get("price_inr", 0) or 0),
            "type": proof.get("type", ""),
            "model_id": proof.get("model_id", ""),
            "model_name": proof.get("model_name", ""),
            "order_id": proof.get("order_id", ""),
            "user_id": user_id,
            "user_name": user_name,
        }
        with open(PAYMENTS_DB, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.warning(f"Failed recording payment: {e}")


def sum_payments_for_date(date_str: str) -> tuple[int, int]:
    """Return (count, total_amount) for lines whose row['date']==date_str."""
    count = 0
    total = 0
    try:
        with open(PAYMENTS_DB, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("date") == date_str:
                    count += 1
                    try:
                        total += int(row.get("amount", 0) or 0)
                    except Exception:
                        pass
    except FileNotFoundError:
        return (0, 0)
    except Exception as e:
        logging.warning(f"Failed reading payments: {e}")
    return (count, total)


async def _finalize_proof(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE', with_photo: bool) -> None:
    proof = context.user_data.pop("awaiting_proof", None)
    context.user_data.pop("awaiting_screenshot", None)
    file_id = context.user_data.pop("screenshot_file_id", None)

    if not proof:
        await update.effective_message.reply_text("No pending proof.")
        return

    # Record to payments file for reporting
    try:
        record_payment(proof, update.effective_user.id, update.effective_user.full_name)
    except Exception as e:
        logging.warning(f"record_payment failed: {e}")

    dest_ids: list[int] = []
    for vid in VERIFIER_CHAT_IDS:
        if vid not in dest_ids:
            dest_ids.append(vid)
    for aid in ADMIN_CHAT_IDS:
        if aid not in dest_ids:
            dest_ids.append(aid)

    lines = [
        "💳 <b>New Payment Proof</b>",
        f"User: <a href='tg://user?id={update.effective_user.id}'>{update.effective_user.full_name}</a>",
        f"Order: <code>{proof['order_id']}</code>",
        f"Item: {proof['model_name']} (₹{proof['price_inr']})",
        f"Unique Transaction Reference number: <code>{proof.get('utr','')}</code>",
    ]

    for dest in dest_ids:
        try:
            if with_photo and file_id:
                await context.bot.send_photo(
                    chat_id=dest,
                    photo=file_id,
                    caption="\n".join(lines),
                    parse_mode=ParseMode.HTML,
                )
            else:
                await context.bot.send_message(
                    chat_id=dest,
                    text="\n".join(lines),
                    parse_mode=ParseMode.HTML,
                )
        except Exception as e:
            logging.warning(f"Failed to forward proof to {dest}: {e}")

    ack = (
        "✅ Thanks! Your payment proof has been submitted for verification.\n"
        "We will connect with you shortly after verification is complete."
    )
    if VERIFIER_HANDLE:
        ack += (
            f"\nFor faster help you can contact {VERIFIER_HANDLE} and share your order ID "
            f"<code>{proof['order_id']}</code>."
        )
    await update.effective_message.reply_html(ack)

# ----------------------- REPORTING ----------------------- #

async def total_cmd(update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
    if not is_admin(update):
        await update.effective_message.reply_text("Admins only.")
        return
    args = (update.effective_message.text or "").split(maxsplit=1)
    if len(args) == 2:
        date_str = args[1].strip()
    else:
        date_str = now_local().strftime("%Y-%m-%d")
    cnt, total = sum_payments_for_date(date_str)
    await update.effective_message.reply_text(
        f"Payments on {date_str}:\nCount: {cnt}\nTotal: ₹{total}")


async def daily_total_job(context: 'ContextTypes.DEFAULT_TYPE') -> None:
    date_str = now_local().strftime("%Y-%m-%d")
    cnt, total = sum_payments_for_date(date_str)
    text = f"📊 Daily total for {date_str}:\nCount: {cnt}\nTotal: ₹{total}"
    for dest in sorted(set(list(ADMIN_CHAT_IDS) + list(VERIFIER_CHAT_IDS))):
        try:
            await context.bot.send_message(chat_id=dest, text=text)
        except Exception as e:
            logging.warning(f"Failed to send daily total to {dest}: {e}")

# ----------------------- ERROR HANDLER ----------------------- #

async def on_error(update: object, context: 'ContextTypes.DEFAULT_TYPE') -> None:  # pragma: no cover
    logging.exception("Unhandled error", exc_info=context.error)
    for aid in ADMIN_CHAT_IDS:
        try:
            await context.bot.send_message(
                chat_id=aid,
                text=f"⚠️ Bot error: {context.error}",
            )
        except Exception:
            pass

# ----------------------- SELFTEST ----------------------- #

def _selftest() -> int:
    link = make_upi_link("d.14360@ptaxis", "Deepa", 2000, "Test Note")
    assert link.startswith("upi://pay?pa=d.14360%40ptaxis")
    assert "pn=Deepa" in link and "am=2000.00" in link and "cu=INR" in link

    cap = build_fixed_caption({"name": "Anika", "desc": "Host"}, 2000, "15 min", link)
    assert "Anika" in cap and "₹2000" in cap and "15 min" in cap and "upi://pay" in cap

    ids = _parse_ids("111, 222 333")
    assert ids == {111, 222, 333}

    # Payments log append + read-back sum
    proof = {"price_inr": 1500, "type": "fixed", "model_id": "x1", "model_name": "Test", "order_id": "ODR-1"}
    record_payment(proof, 1, "Tester")
    today = now_local().strftime("%Y-%m-%d")
    cnt, total = sum_payments_for_date(today)
    assert isinstance(cnt, int) and isinstance(total, int)
    return 0

# ----------------------- HEALTH SERVER (polling mode) ----------------------- #

def maybe_start_health_server() -> None:
    """Start a tiny HTTP server on $PORT (if set) so web hosts see a healthy endpoint."""
    port_str = _getenv("PORT", "")
    if not port_str:
        return
    try:
        port = int(port_str)
    except Exception:
        logging.warning(f"Invalid PORT env value '{port_str}' — skipping health server")
        return

    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        def log_message(self, format, *args):
            return

    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        logging.info(f"Health server listening on 0.0.0.0:{port}")
    except Exception as e:
        logging.warning(f"Failed to start health server: {e}")

# ----------------------- MAIN ----------------------- #

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == "selftest":
        rc = _selftest()
        print("Selftest OK")
        raise SystemExit(rc)

    if not TELEGRAM_AVAILABLE:
        print(
            "python-telegram-bot is not installed.\n"
            "Install deps: pip install 'python-telegram-bot[webhooks]' qrcode pillow httpx aiohttp tzdata"
        )
        raise SystemExit(1)

    if not BOT_TOKEN:
        raise SystemExit("BOT_TOKEN is not set")

    ensure_default_files()

    mode = "webhook" if (USE_WEBHOOK and WEBHOOK_URL) else "polling"
    logging.info(f"Starting bot… (mode: {mode})")
    logging.info(f"Base dir: {BASE_DIR}")
    logging.info(f"Resolved MODELS_DB: {MODELS_DB}")
    logging.info(f"Resolved FIXED_DB:  {FIXED_DB}")
    logging.info(f"Join required: {REQUIRE_JOIN}; Channel: {CHANNEL_USERNAME}")
    logging.info(f"Admin chat ids: {sorted(ADMIN_CHAT_IDS)}; verifier ids: {sorted(VERIFIER_CHAT_IDS)}")

    app: 'Application' = ApplicationBuilder().token(BOT_TOKEN).build()

    # commands (user)
    app.add_handler(CommandHandler("start", start, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("help", help_cmd, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("prices", prices_cmd, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("fixedmodels", fixedmodels_cmd, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("reload", reload_cmd, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("whoami", lambda u, c: u.effective_message.reply_text(str(u.effective_user.id)), filters.ChatType.PRIVATE))

    # commands (admin)
    app.add_handler(CommandHandler("fixedadd", fixed_add_cmd, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("fixeddel", fixed_del_cmd, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("setfixedprice", set_fixed_price_cmd, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("fixedlist", fixed_list_cmd, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("total", total_cmd, filters.ChatType.PRIVATE))

    # callbacks & messages
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & filters.TEXT, on_text))
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & filters.PHOTO, on_photo))

    # error handler
    app.add_error_handler(on_error)

    load_models()
    fdata = load_fixed()
    logging.info(f"Loaded plans: {len(MODELS)} | fixed models: {len(fdata['models'])}")

    # schedule daily total if configured and JobQueue exists
    if DAILY_SUM_HHMM:
        jq = getattr(app, "job_queue", None)
        if jq:
            try:
                hh, mm = DAILY_SUM_HHMM.strip().split(":")
                t = dtime(hour=int(hh), minute=int(mm), tzinfo=ZONE)
                jq.run_daily(daily_total_job, time=t, name="daily-total")
                logging.info(f"Daily total scheduled at {DAILY_SUM_HHMM} {TIMEZONE}")
            except Exception as e:
                logging.warning(f"Failed to schedule daily total: {e}")
        else:
            logging.warning("JobQueue not available — install PTB with job-queue extra to enable daily totals")

    if USE_WEBHOOK and WEBHOOK_URL:
        # Webhook mode
        url_path = BOT_TOKEN  # secret path
        full_webhook = f"{WEBHOOK_URL.rstrip('/')}/{url_path}"
        logging.info(f"Webhook -> {full_webhook}")
        app.run_webhook(listen="0.0.0.0", port=PORT, url_path=url_path, webhook_url=full_webhook, secret_token=SECRET_TOKEN)
    else:
        # Polling mode (also start tiny health server if PORT set)
        maybe_start_health_server()
        logging.info("Bot started. Press Ctrl+C to stop.")

        # --- Webhook vs Polling start ---
        USE_WEBHOOK = os.getenv("USE_WEBHOOK", "0") in {"1", "true", "True", "yes", "YES"}
        WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip().rstrip("/")
        WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip() or None
        PORT = int(os.getenv("PORT", "10000"))

        USE_WEBHOOK = os.getenv("USE_WEBHOOK", "0") in {"1", "true", "True", "yes", "YES"}
        WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip().rstrip("/")
        WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip() or None
        PORT = int(os.getenv("PORT", "10000"))

        if USE_WEBHOOK:
            if not WEBHOOK_URL:
                raise SystemExit("WEBHOOK_URL is required when USE_WEBHOOK=1")
            url_path = BOT_TOKEN
            public_url = f"{WEBHOOK_URL}/{url_path}"
            logging.info(f"Webhook -> {public_url}")
            app.run_webhook(
                listen="0.0.0.0",
                port=PORT,
                url_path=url_path,
                webhook_url=public_url,
                secret_token=WEBHOOK_SECRET,
            )
        else:
            app.run_polling()
        # --- Webhook vs Polling end ---
if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped")
