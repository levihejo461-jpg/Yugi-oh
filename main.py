import os
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
import filetype

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Bot is alive, Captain Levi!")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo = update.message.photo[-1]
    file = await photo.get_file()
    os.makedirs("downloads", exist_ok=True)
    file_path = os.path.join("downloads", f"{file.file_id}.jpg")
    await file.download_to_drive(file_path)

    kind = filetype.guess(file_path)
    msg = f"Image type: {kind.mime}" if kind else "Unknown image type."
    await update.message.reply_text(msg)

async def main():
    TOKEN = "YOUR_BOT_TOKEN_HERE"
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
