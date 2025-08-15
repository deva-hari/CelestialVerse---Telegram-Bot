# bot.py
import os
import json
import hashlib
import asyncio
from datetime import datetime, UTC
from typing import List, Tuple, Optional
from io import BytesIO

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Google Generative AI (Gemini)
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env from the workspace root, fallback to current file location
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
load_dotenv(override=True)

# ===================== CONFIG =====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", None)
if not TELEGRAM_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN is not set. Please check your .env file and environment variables."
    )

ADMIN_USER_ID = os.getenv("TELEGRAM_ADMIN_ID", "").strip()  # optional, e.g. "123456789"

# Global default (admin-provided) Gemini key; users can override with /set_api_key
GLOBAL_GENAI_API_KEY = os.getenv("GENAI_API_KEY", "").strip()

DEFAULT_MODEL = os.getenv("GENAI_MODEL", "gemini-2.5-flash").strip()
DEFAULT_PROMPT = "Translate the following Chinese text into fluent English while keeping the literary tone."

# Storage
CONFIG_FILE = "configs.json"  # per-user config (api_key, model, prompt)
HASH_DB_FILE = "file_hashes.json"  # content_hash -> translated_path
HISTORY_FILE = "history.json"  # per-user history listing
RECEIVED_DIR = "received_files"
TRANSLATED_DIR = "translated_files"
os.makedirs(RECEIVED_DIR, exist_ok=True)
os.makedirs(TRANSLATED_DIR, exist_ok=True)

# Telegram limits
MAX_MESSAGE_LEN = 4096
SAFE_SLICE = 4090

# Models menu (extend freely)
AVAILABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]


# ===================== UTIL: FILE I/O =====================
def _load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_configs():
    return _load_json(CONFIG_FILE, {})


def save_configs(configs):
    _save_json(CONFIG_FILE, configs)


def get_user_config(user_id: str):
    configs = load_configs()
    cfg = configs.get(user_id, {})
    return {
        "api_key": cfg.get("api_key", ""),  # empty means "use global"
        "model": cfg.get("model", DEFAULT_MODEL),
        "prompt": cfg.get("prompt", DEFAULT_PROMPT),
    }


def set_user_config(user_id: str, key: str, value: str):
    configs = load_configs()
    if user_id not in configs:
        configs[user_id] = {}
    configs[user_id][key] = value
    save_configs(configs)


def reset_user_config(user_id: str):
    configs = load_configs()
    configs[user_id] = {}  # back to defaults (uses global/defaults)
    save_configs(configs)


def mask_key(k: str) -> str:
    if not k:
        return "(not set)"
    if len(k) <= 8:
        return "*" * len(k)
    return k[:4] + "*" * (len(k) - 8) + k[-4:]


# ===================== UTIL: ENCODING & CHUNKING =====================
def safe_read_text(path: str) -> str:
    encodings = [
        "utf-8",
        "utf-16",
        "utf-16le",
        "utf-16be",
        "gb18030",
        "big5",
        "latin-1",
    ]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    # last resort: binary decode ignoring errors
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")


def chunk_text_by_chars(text: str, max_chars: int = 6000) -> List[str]:
    # Character-based chunking with paragraph awareness
    parts = []
    current = []
    size = 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if size + len(p) + 2 > max_chars and current:
            parts.append("\n\n".join(current))
            current = [p]
            size = len(p) + 2
        else:
            current.append(p)
            size += len(p) + 2
    if current:
        parts.append("\n\n".join(current))
    return parts if parts else [""]


def split_for_telegram(text: str) -> List[str]:
    return [text[i : i + SAFE_SLICE] for i in range(0, len(text), SAFE_SLICE)] or [""]


# ===================== GEMINI =====================
def _get_effective_api_key(user_cfg: dict) -> Optional[str]:
    user_key = (user_cfg or {}).get("api_key", "").strip()
    if user_key:
        return user_key
    if GLOBAL_GENAI_API_KEY:
        return GLOBAL_GENAI_API_KEY
    return None


async def _gemini_generate(
    model_name: str, prompt: str, text: str, api_key: str
) -> str:
    # Run blocking Gemini call in a worker thread
    def _call():
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(f"{prompt}\n\n{text}")
        if not resp or not hasattr(resp, "text") or not resp.text:
            return ""
        return resp.text.strip()

    return await asyncio.to_thread(_call)


async def translate_with_gemini(text: str, cfg: dict) -> str:
    api_key = _get_effective_api_key(cfg)
    if not api_key:
        return "⚠️ Gemini API key not set. Ask the bot admin to set a global key or use /set_api_key <your_key>."
    model_name = cfg.get("model", DEFAULT_MODEL)
    prompt = cfg.get("prompt", DEFAULT_PROMPT)

    # Chunk long input
    chunks = chunk_text_by_chars(text, max_chars=6000)
    outputs: List[str] = []
    for i, ch in enumerate(chunks, 1):
        try:
            out = await _gemini_generate(model_name, prompt, ch, api_key)
            outputs.append(out if out else "")
        except Exception as e:
            outputs.append(f"[Chunk {i} error] {e}")
    return "\n\n".join(outputs).strip()


# ===================== HISTORY =====================
def add_history(user_id: str, original_name: str, translated_path: str):
    hist = _load_json(HISTORY_FILE, {})
    user_list = hist.get(user_id, [])
    user_list.append(
        {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "original_name": original_name,
            "translated_path": translated_path,
        }
    )
    # keep last 20
    hist[user_id] = user_list[-20:]
    _save_json(HISTORY_FILE, hist)


def get_history(user_id: str) -> List[dict]:
    hist = _load_json(HISTORY_FILE, {})
    return hist.get(user_id, [])


# ===================== TELEGRAM HANDLERS =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome! Send me Chinese text or a .txt file and I'll translate it.\n\n"
        "Commands:\n"
        "/set_prompt <text>\n"
        "/set_api_key <key>\n"
        "/set_model – choose a model\n"
        "/get_config – view your current settings\n"
        "/reset_config – reset to defaults\n"
        "/history – re-download recent translations"
    )


async def set_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not context.args:
        await update.message.reply_text("Usage: /set_prompt <your custom prompt>")
        return
    prompt = " ".join(context.args).strip()
    set_user_config(user_id, "prompt", prompt)
    await update.message.reply_text("✅ Prompt updated.")


async def set_api_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not context.args:
        await update.message.reply_text("Usage: /set_api_key <your_api_key>")
        return
    key = context.args[0].strip()
    set_user_config(user_id, "api_key", key)
    await update.message.reply_text("✅ API key updated (stored per-user).")


async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton(m, callback_data=f"model:{m}")] for m in AVAILABLE_MODELS
    ]
    keyboard.append(
        [
            InlineKeyboardButton(
                "Custom (send name next message)", callback_data="model:custom"
            )
        ]
    )
    await update.message.reply_text(
        "Choose a model:", reply_markup=InlineKeyboardMarkup(keyboard)
    )
    # flag we’re awaiting a custom model only after the user taps that button


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)

    if query.data.startswith("model:"):
        model = query.data.split(":", 1)[1]
        if model == "custom":
            context.user_data["awaiting_custom_model"] = True
            await query.edit_message_text(
                "Send me the model name (e.g., gemini-2.5-flash):"
            )
        else:
            set_user_config(user_id, "model", model)
            context.user_data["awaiting_custom_model"] = False
            await query.edit_message_text(f"✅ Model updated to `{model}`")

    elif query.data.startswith("history:"):
        # data: history:<index>
        try:
            idx = int(query.data.split(":", 1)[1])
            items = get_history(user_id)
            if 0 <= idx < len(items):
                path = items[idx]["translated_path"]
                if os.path.exists(path):
                    await context.bot.send_document(
                        chat_id=query.message.chat_id,
                        document=InputFile(path),
                        filename="translation.txt",
                    )
                else:
                    await query.message.reply_text("⚠️ File no longer exists on server.")
            else:
                await query.message.reply_text("⚠️ Invalid history item.")
        except Exception as e:
            await query.message.reply_text(f"⚠️ History error: {e}")


async def handle_custom_model_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Only treat as custom-model input if we're awaiting it
    if context.user_data.get("awaiting_custom_model"):
        user_id = str(update.effective_user.id)
        model = (update.message.text or "").strip()
        if model:
            set_user_config(user_id, "model", model)
            context.user_data["awaiting_custom_model"] = False
            await update.message.reply_text(f"✅ Custom model set to `{model}`")
        else:
            await update.message.reply_text("⚠️ Empty model name. Try again.")
        return True
    return False


async def get_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    cfg = get_user_config(user_id)
    effective_key = _get_effective_api_key(cfg)
    key_source = (
        "user"
        if cfg.get("api_key")
        else ("global" if GLOBAL_GENAI_API_KEY else "not set")
    )
    await update.message.reply_text(
        "🔧 Your configuration:\n"
        f"- Model: {cfg.get('model', DEFAULT_MODEL)}\n"
        f"- Prompt: {cfg.get('prompt', DEFAULT_PROMPT)[:200]}{'...' if len(cfg.get('prompt', DEFAULT_PROMPT))>200 else ''}\n"
        f"- API key ({key_source}): {mask_key(effective_key or '')}"
    )


async def reset_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    reset_user_config(user_id)
    await update.message.reply_text(
        "✅ Config reset to defaults (uses global API key if set)."
    )


async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    items = get_history(user_id)
    if not items:
        await update.message.reply_text("No history yet.")
        return
    # show last 10 with buttons
    buttons = []
    start = max(0, len(items) - 10)
    for i in range(start, len(items)):
        it = items[i]
        caption = f"{i}. {it['original_name']} ({it['ts']})"
        buttons.append([InlineKeyboardButton(caption, callback_data=f"history:{i}")])
    await update.message.reply_text(
        "Recent translations (tap to download):",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


# ======= TEXT MESSAGES =======
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If awaiting custom model name, handle that
    if await handle_custom_model_text(update, context):
        return
    user_id = str(update.effective_user.id)
    cfg = get_user_config(user_id)
    original_text = update.message.text or ""
    await update.message.chat.send_action(ChatAction.TYPING)
    # Save input text to file
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{user_id}_{now_str}_chat.txt"
    received_path = os.path.join(RECEIVED_DIR, file_name)
    with open(received_path, "w", encoding="utf-8") as f:
        f.write(original_text)
    # Hash check
    content_hash = hashlib.sha256(
        original_text.encode("utf-8", errors="ignore")
    ).hexdigest()
    hash_db = _load_json(HASH_DB_FILE, {})
    if content_hash in hash_db and os.path.exists(hash_db[content_hash]):
        await update.message.reply_text(
            "⚠️ This text was translated before. Sending previous translation."
        )
        with open(hash_db[content_hash], "rb") as f:
            prev_bytes = f.read()
        bio = BytesIO(prev_bytes)
        bio.name = "translation.txt"
        bio.seek(0)
        await update.message.reply_document(document=bio, filename="translation.txt")
        add_history(user_id, "chat.txt", hash_db[content_hash])
        return
    # Translate
    translated = await translate_with_gemini(original_text, cfg)
    # Save translated text to file
    translated_file_name = f"{user_id}_{now_str}_chat.txt"
    output_path = os.path.join(TRANSLATED_DIR, translated_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)
    # Update hash db
    hash_db[content_hash] = output_path
    _save_json(HASH_DB_FILE, hash_db)
    add_history(user_id, "chat.txt", output_path)
    # Send as message if short, else as file
    if len(translated) < 4000:
        await update.message.reply_text(translated)
    else:
        bio = BytesIO(translated.encode("utf-8"))
        bio.name = "translation.txt"
        bio.seek(0)
        await update.message.reply_document(document=bio, filename="translation.txt")


# ======= FILE MESSAGES (.txt) =======
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    file_name = document.file_name
    # ✅ Only allow .txt files
    if not file_name.lower().endswith(".txt"):
        await update.message.reply_text("⚠️ Only .txt files are supported.")
        return
    # Download file
    file = await document.get_file()
    received_file_name = f"{update.effective_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_name}"
    file_path = os.path.join(RECEIVED_DIR, received_file_name)
    await file.download_to_drive(file_path)
    # ✅ Try multiple encodings
    encodings = ["utf-8", "utf-16", "gbk", "latin-1"]
    file_content = None
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                file_content = f.read()
            break
        except Exception:
            continue
    if not file_content:
        await update.message.reply_text("❌ Could not read file. Unsupported encoding.")
        return
    # Hash check before translation
    content_hash = hashlib.sha256(
        file_content.encode("utf-8", errors="ignore")
    ).hexdigest()
    hash_db = _load_json(HASH_DB_FILE, {})
    if content_hash in hash_db and os.path.exists(hash_db[content_hash]):
        await update.message.reply_text(
            "⚠️ This file was translated before. Sending previous translation."
        )
        # Send previous translation as BytesIO
        with open(hash_db[content_hash], "rb") as f:
            prev_bytes = f.read()
        bio = BytesIO(prev_bytes)
        bio.name = file_name
        bio.seek(0)
        await update.message.reply_document(document=bio, filename=file_name)
        add_history(str(update.effective_user.id), file_name, hash_db[content_hash])
        return
    # Respond immediately
    await update.message.reply_text("⏳ File received! Translating...")
    # Split into safe chunks (Telegram limit ~4096 chars)
    chunks = [file_content[i : i + 3500] for i in range(0, len(file_content), 3500)]
    translated_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        try:
            translated = await translate_with_gemini(
                chunk, get_user_config(str(update.effective_user.id))
            )
            translated_chunks.append(translated)
        except Exception as e:
            translated_chunks.append(f"[Error translating chunk {idx}: {e}]")
    final_translation = "\n\n".join(translated_chunks)
    # Save to file in translated_files with original name
    translated_file_name = f"{update.effective_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_name}"
    output_path = os.path.join(TRANSLATED_DIR, translated_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_translation)
    # Update hash db
    hash_db[content_hash] = output_path
    _save_json(HASH_DB_FILE, hash_db)
    # Send back translated file as BytesIO to ensure .txt type
    bio = BytesIO(final_translation.encode("utf-8"))
    bio.name = file_name
    bio.seek(0)
    await update.message.reply_document(document=bio, filename=file_name)
    add_history(str(update.effective_user.id), file_name, output_path)


async def _process_file_translation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str,
    original_name: str,
    content: str,
    translated_path: str,
    content_hash: str,
):
    try:
        cfg = get_user_config(user_id)
        result = await translate_with_gemini(content, cfg)
        # Always save as .txt with original filename
        base_txt_name = os.path.splitext(original_name)[0] + ".txt"
        final_path = os.path.join(
            TRANSLATED_DIR,
            f"{user_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{base_txt_name}",
        )
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(result)
        # Update hash db
        hash_db = _load_json(HASH_DB_FILE, {})
        hash_db[content_hash] = final_path
        _save_json(HASH_DB_FILE, hash_db)
        # Add history
        add_history(user_id, original_name, final_path)
        # Send as BytesIO to ensure Telegram treats as text
        bio = BytesIO(result.encode("utf-8"))
        bio.name = (
            original_name
            if original_name.lower().endswith(".txt")
            else original_name + ".txt"
        )
        bio.seek(0)
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=bio,
            filename=bio.name,
            caption="✅ Done.",
        )
    except Exception as e:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=f"⚠️ Translation failed: {e}"
        )


# ===================== ADMIN (optional) =====================
async def set_global_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_USER_ID or str(update.effective_user.id) != ADMIN_USER_ID:
        await update.message.reply_text("⛔ Not authorized.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /admin_set_global_key <key>")
        return
    key = context.args[0].strip()
    # Persisting env at runtime isn't trivial; we simulate by saving to a special user 'GLOBAL'
    set_user_config("GLOBAL", "api_key", key)
    # And read it into process global for use immediately
    global GLOBAL_GENAI_API_KEY
    GLOBAL_GENAI_API_KEY = key
    await update.message.reply_text(
        "✅ Global API key updated for all users (unless they set their own)."
    )


# ===================== MAIN =====================
def build_application() -> Application:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("set_prompt", set_prompt))
    app.add_handler(CommandHandler("set_api_key", set_api_key))
    app.add_handler(CommandHandler("set_model", set_model))
    app.add_handler(CommandHandler("get_config", get_config))
    app.add_handler(CommandHandler("reset_config", reset_config))
    app.add_handler(CommandHandler("history", history_cmd))
    app.add_handler(CommandHandler("admin_set_global_key", set_global_key))

    # Callback buttons
    app.add_handler(CallbackQueryHandler(button))

    # Messages
    # Order matters: custom-model capture happens only if awaiting flag is set.
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    return app


def main():
    app = build_application()

    # Choose between long-polling and webhook via env
    mode = os.getenv("RUN_MODE", "polling").lower()
    if mode == "webhook":
        port = int(os.getenv("PORT", "8080"))
        webhook_url = os.getenv(
            "WEBHOOK_URL", ""
        ).strip()  # e.g. https://your-app.onrender.com/webhook
        if not webhook_url:
            raise RuntimeError("WEBHOOK_URL must be set for webhook mode.")
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path="webhook",
            webhook_url=webhook_url.rstrip("/") + "/webhook",
        )
    else:
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
