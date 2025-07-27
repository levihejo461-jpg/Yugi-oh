from telegram.ext import Updater, CommandHandler

def start(update, context):
    update.message.reply_text("Bot is alive, Captain Levi!")

def main():
    TOKEN = "YOUR_BOT_TOKEN_HERE"
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
