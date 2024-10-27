import logging
import joblib
from scipy.sparse import hstack
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Load the model and vectorizers
model = joblib.load('fake_news_model.pkl')
title_vectorizer = joblib.load('title_vectorizer.pkl')
text_vectorizer = joblib.load('text_vectorizer.pkl')

# Enable logging for debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Function to predict if a news article is real or fake
def predict_news(article):
    X_title = title_vectorizer.transform([article])
    X_text = text_vectorizer.transform([article])
    X = hstack([X_title, X_text])
    prediction = model.predict(X)
    return "Real" if prediction[0] == 0 else "Fake"

# Command handler to start the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome! Send me a news article, and I'll tell you if it's likely real or fake.")

# Message handler to process the input article
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    article = update.message.text
    prediction = predict_news(article)
    response = f"The news article is likely: {prediction}"
    await update.message.reply_text(response)

# Error handler
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.warning('Update "%s" caused error "%s"', update, context.error)

# Main function to set up the bot
def main():
    # Replace 'YOUR_TELEGRAM_BOT_TOKEN' with your actual bot token from BotFather
    application = ApplicationBuilder().token('7830354209:AAFztAmGuDK_Ejhb_q1_9JK7GgAiPP4FHFw').build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
