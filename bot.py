import os
import json
import hashlib
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
import google.generativeai as genai

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "8251111633:AAGsVnpCmgOsCS2XJZ0C5Z5U65EtSb1Zt2w"
)
CONFIG_FILE = "configs.json"
DEFAULT_API_KEY = os.getenv(
    "GENAI_API_KEY", "GEMINI_API_KEY"
)  # Replace with your real key
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_PROMPT = "Translate the following Chinese text into fluent English while keeping the literary tone."


# ------------------ CONFIG MANAGEMENT ------------------
def load_configs():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_configs(configs):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)


def get_user_config(user_id: str):
    configs = load_configs()
    return configs.get(
        user_id,
        {"api_key": DEFAULT_API_KEY, "model": DEFAULT_MODEL, "prompt": DEFAULT_PROMPT},
    )


def set_user_config(user_id: str, key: str, value: str):
    configs = load_configs()
    if user_id not in configs:
        configs[user_id] = {}
    configs[user_id][key] = value
    save_configs(configs)


# ------------------ GEMINI TRANSLATION ------------------
async def translate_with_gemini(text: str, config: dict) -> str:
    api_key = config.get("api_key", DEFAULT_API_KEY)
    if not api_key or api_key == "GEMINI_API_KEY":
        return "⚠️ Gemini API key not set. Use /set_api_key <your_api_key> to set it."
    try:
        genai.configure(api_key=api_key)
        model_name = config.get("model", DEFAULT_MODEL)
        prompt = config.get("prompt", DEFAULT_PROMPT)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(f"{prompt}\n\n{text}")
        return (
            response.text.strip()
            if response and hasattr(response, "text")
            else "⚠️ No response from model."
        )
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# ------------------ COMMAND HANDLERS ------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome! Send me Chinese text or a .txt file, and I'll translate it."
    )


async def set_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not context.args:
        await update.message.reply_text("Usage: /set_prompt <your custom prompt>")
        return
    prompt = " ".join(context.args)
    set_user_config(user_id, "prompt", prompt)
    await update.message.reply_text(f"✅ Prompt updated:\n{prompt}")


async def set_api_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not context.args:
        await update.message.reply_text("Usage: /set_api_key <your_api_key>")
        return
    key = context.args[0]
    set_user_config(user_id, "api_key", key)
    await update.message.reply_text("✅ API key updated!")


async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton(
                "Gemini 1.5 Flash", callback_data="model:gemini-1.5-flash"
            )
        ],
        [InlineKeyboardButton("Gemini 1.5 Pro", callback_data="model:gemini-1.5-pro")],
        [
            InlineKeyboardButton(
                "Gemini 2.0 Flash", callback_data="model:gemini-2.0-flash"
            )
        ],
        [
            InlineKeyboardButton(
                "Gemini 2.5 Flash", callback_data="model:gemini-2.5-flash"
            )
        ],
        [InlineKeyboardButton("Custom (send name)", callback_data="model:custom")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose a model:", reply_markup=reply_markup)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)
    if query.data.startswith("model:"):
        model = query.data.split(":", 1)[1]
        if model == "custom":
            await query.edit_message_text(
                "Send me the model name (e.g., `gemini-2.5-flash`):"
            )
            context.user_data["awaiting_custom_model"] = True
        else:
            set_user_config(user_id, "model", model)
            await query.edit_message_text(f"✅ Model updated to `{model}`")


async def handle_custom_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("awaiting_custom_model"):
        user_id = str(update.effective_user.id)
        model = update.message.text.strip()
        set_user_config(user_id, "model", model)
        context.user_data["awaiting_custom_model"] = False
        await update.message.reply_text(f"✅ Custom model set to `{model}`")


# ------------------ MESSAGE HANDLERS ------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    config = get_user_config(user_id)
    original_text = update.message.text
    translated = await translate_with_gemini(original_text, config)
    # Handle Telegram message size limits (split if > 4096 chars)
    for i in range(0, len(translated), 4000):
        await update.message.reply_text(translated[i : i + 4000])


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc.file_name.endswith(".txt"):
        await update.message.reply_text("⚠️ Only .txt files are supported.")
        return
    user_id = str(update.effective_user.id)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create folder paths
    received_dir = os.path.join("received_files")
    translated_dir = os.path.join("translated_files")
    os.makedirs(received_dir, exist_ok=True)
    os.makedirs(translated_dir, exist_ok=True)
    # Unique file name
    base_filename = f"{user_id}_{now_str}_{doc.file_name}"
    received_path = os.path.join(received_dir, base_filename)
    translated_path = os.path.join(translated_dir, base_filename)
    # Download file
    file = await doc.get_file()
    await file.download_to_drive(received_path)
    with open(received_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Hash content
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    # Check for duplicate
    hash_db_path = "file_hashes.json"
    if os.path.exists(hash_db_path):
        with open(hash_db_path, "r", encoding="utf-8") as f:
            hash_db = json.load(f)
    else:
        hash_db = {}
    if content_hash in hash_db:
        await update.message.reply_text(
            "⚠️ This file has already been translated. Sending previous translation."
        )
        prev_translated_path = hash_db[content_hash]
        await update.message.reply_document(
            open(prev_translated_path, "rb"), filename="translation.txt"
        )
        return
    config = get_user_config(user_id)
    translated = await translate_with_gemini(content, config)
    with open(translated_path, "w", encoding="utf-8") as f:
        f.write(translated)
    # Store hash and path
    hash_db[content_hash] = translated_path
    with open(hash_db_path, "w", encoding="utf-8") as f:
        json.dump(hash_db, f, indent=2)
    await update.message.reply_document(
        open(translated_path, "rb"), filename="translation.txt"
    )


# ------------------ MAIN ------------------
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("set_prompt", set_prompt))
    app.add_handler(CommandHandler("set_api_key", set_api_key))
    app.add_handler(CommandHandler("set_model", set_model))
    app.add_handler(CallbackQueryHandler(button))
    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(
        MessageHandler(filters.TEXT & filters.Regex(".*"), handle_custom_model)
    )
    app.run_polling()


if __name__ == "__main__":
    main()
