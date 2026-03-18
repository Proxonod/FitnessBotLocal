"""
Telegram Bot: Handles incoming messages and routes to Orchestrator.

Supports:
  - Photo messages (with optional caption)
  - Text messages
  - Commands (/start, /today, /stats, /products)
"""

import io
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from src.orchestrator import Orchestrator
from src.config import TELEGRAM_BOT_TOKEN


orchestrator = Orchestrator()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    response = await orchestrator.handle_text(
        telegram_id=user.id,
        user_name=user.first_name,
        text="/start",
    )
    await update.message.reply_text(response, parse_mode="Markdown")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages — food recognition or label OCR."""
    user = update.effective_user
    caption = update.message.caption

    # Get the largest photo version
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    # Download to memory
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    photo_bytes = buf.getvalue()

    print(f"📸 Photo from {user.first_name} ({user.id}), "
          f"caption: {caption or 'none'}, size: {len(photo_bytes)} bytes")

    # Send "typing" indicator while processing
    await update.message.chat.send_action("typing")

    response = await orchestrator.handle_photo(
        telegram_id=user.id,
        user_name=user.first_name,
        photo_bytes=photo_bytes,
        caption=caption,
    )
    await update.message.reply_text(response, parse_mode="Markdown")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages."""
    user = update.effective_user
    text = update.message.text

    print(f"💬 Text from {user.first_name} ({user.id}): {text[:80]}...")

    await update.message.chat.send_action("typing")

    response = await orchestrator.handle_text(
        telegram_id=user.id,
        user_name=user.first_name,
        text=text,
    )
    await update.message.reply_text(response, parse_mode="Markdown")


def create_bot() -> Application:
    """Create and configure the Telegram bot."""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", lambda u, c: _cmd(u, c, "/today")))
    app.add_handler(CommandHandler("stats", lambda u, c: _cmd(u, c, "/stats")))
    app.add_handler(CommandHandler("products", lambda u, c: _cmd(u, c, "/products")))

    # Message handlers
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    return app


async def _cmd(update: Update, context: ContextTypes.DEFAULT_TYPE, cmd: str):
    """Generic command handler."""
    user = update.effective_user
    response = await orchestrator.handle_text(user.id, user.first_name, cmd)
    await update.message.reply_text(response, parse_mode="Markdown")
