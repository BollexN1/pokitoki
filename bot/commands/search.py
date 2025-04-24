from telegram import Update
from telegram.ext import ContextTypes
from bot.config import Config
from bot.ai.chat import Chat
from bot.models import get_model

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    chat: Chat = context.bot_data["chat"]
    chat_id = str(update.effective_chat.id)
    
    if not _is_authorized(update, config):
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Please provide a search query.")
        return
    
    model = get_model(chat_id)
    try:
        message = await update.message.reply_text("🔍 Начинаю поиск...")
        text_buffer = ""
        for chunk in chat.search(chat_id, query, model):
            if chunk["type"] == "thinking":
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message.message_id,
                    text=f"🔍 Поиск: {chunk['content']}\n\nТекущий ответ: {text_buffer}"
                )
            elif chunk["type"] == "text":
                text_buffer += chunk["content"]
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message.message_id,
                    text=f"🔍 Поиск завершен:\n\n{text_buffer}"
                )
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

def _is_authorized(update: Update, config: Config) -> bool:
    chat_id = str(update.effective_chat.id)
    username = update.effective_user.username
    return (
        not config.usernames or username in config.usernames
    ) and (
        not config.chat_ids or chat_id in config.chat_ids
    )
