import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Your /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot is working, Captain Levi!")

# Main function
async def main():
    # Replace 'YOUR_BOT_TOKEN' with your real bot token
    application = ApplicationBuilder().token("YOUR_BOT_TOKEN").build()

    # Add command handler
    application.add_handler(CommandHandler("start", start))

    # Start the bot
    await application.run_polling()

# Run the bot
if __name__ == "__main__":
    asyncio.run(main())
