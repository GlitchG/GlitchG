"""Telegram bot: forward a voice message / audio / video, get a transcript with speaker roles.

Environment variables:
  TELEGRAM_BOT_TOKEN   required - from @BotFather. Never commit it.
  ALLOWED_USER_IDS     recommended - comma-separated Telegram user ids allowed to use the bot
                       (send /start to see yours). Unset = anyone who finds the bot can use your GPU.
  ANTHROPIC_API_KEY    optional - enables automatic role detection with Claude.

Run:  python bot.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from diarize_transcribe.cli import parse_roles
from diarize_transcribe.formatters import fmt_clock, render
from diarize_transcribe.pipeline import load_asr, load_diarizer, run
from diarize_transcribe.roles import guess_roles

log = logging.getLogger("bot")

# Bots on the public Telegram API can only download files up to 20 MB.
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
TELEGRAM_TEXT_LIMIT = 4096
# Longer transcripts are sent as a file instead of a wall of messages.
INLINE_TEXT_LIMIT = 3500

ALLOWED_USER_IDS = {int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}
AUTO_ROLES_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))

HELP = (
    "Send me a voice message, audio file or video (up to 20 MB) and I'll reply with "
    "a transcript that shows who said what.\n\n"
    "Naming speakers (in the order they first speak):\n"
    "• add a caption with commas, e.g. Manager, Client\n"
    "• or a caption describing the call (e.g. sales call with a hotel owner) to help auto roles\n"
    "• or set defaults for this chat: /roles Manager, Client\n"
    "• /roles with nothing after it clears them\n"
    "• /auto toggles automatic role detection with Claude"
    + ("" if AUTO_ROLES_AVAILABLE else " (needs ANTHROPIC_API_KEY on the server)")
    + "\n• /format txt|md|srt|json sets the file format"
)

# One GPU: process one recording at a time, others wait in line.
gpu_lock = asyncio.Lock()


def allowed(update: Update) -> bool:
    user = update.effective_user
    return not ALLOWED_USER_IDS or (user is not None and user.id in ALLOWED_USER_IDS)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    text = f"Hi! Your Telegram user id is {user.id}.\n\n"
    if not allowed(update):
        text += "You're not on this bot's allow-list. Ask the owner to add your id."
    else:
        text += HELP
    await update.message.reply_text(text)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


async def roles_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    roles = parse_roles(" ".join(context.args)) if context.args else None
    context.chat_data["roles"] = roles
    await update.message.reply_text(f"Default roles: {', '.join(roles)}" if roles else "Default roles cleared.")


async def auto_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    if not AUTO_ROLES_AVAILABLE:
        await update.message.reply_text("Auto roles need ANTHROPIC_API_KEY set where the bot runs.")
        return
    new = not context.chat_data.get("auto", True)
    context.chat_data["auto"] = new
    await update.message.reply_text(f"Automatic role detection: {'on' if new else 'off'}")


async def format_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    fmt = (context.args[0].lower() if context.args else "").strip()
    if fmt not in ("txt", "md", "srt", "json"):
        await update.message.reply_text("Usage: /format txt|md|srt|json")
        return
    context.chat_data["format"] = fmt
    await update.message.reply_text(f"File format: {fmt}")


def _pick_media(update: Update):
    """Return (telegram file object, file name) for any audio-like attachment."""
    m = update.message
    if m.voice:
        return m.voice, "voice.ogg"
    if m.audio:
        return m.audio, m.audio.file_name or "audio.mp3"
    if m.video_note:
        return m.video_note, "video_note.mp4"
    if m.video:
        return m.video, m.video.file_name or "video.mp4"
    if m.document:
        return m.document, m.document.file_name or "file"
    return None, None


def _chunks(text: str, size: int = TELEGRAM_TEXT_LIMIT):
    """Split on paragraph boundaries so turns aren't cut mid-sentence."""
    buf = ""
    for para in text.split("\n\n"):
        piece = para if not buf else "\n\n" + para
        if len(buf) + len(piece) > size and buf:
            yield buf
            buf = para
        else:
            buf += piece
        while len(buf) > size:
            yield buf[:size]
            buf = buf[size:]
    if buf:
        yield buf


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        await update.message.reply_text("Sorry, this bot is private. Send /start to see your user id.")
        return

    media, name = _pick_media(update)
    if media is None:
        return
    if media.file_size and media.file_size > MAX_DOWNLOAD_BYTES:
        await update.message.reply_text(
            f"This file is {media.file_size / 1024 / 1024:.0f} MB, but Telegram only lets bots download "
            "files up to 20 MB. Compress it (e.g. export as a low-bitrate mp3/ogg), split it, or use the web app."
        )
        return

    status = await update.message.reply_text("⏳ Got it, transcribing…")
    if gpu_lock.locked():
        await status.edit_text("⏳ Another recording is being processed. Yours is next in line…")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / Path(name).name
            tg_file = await context.bot.get_file(media.file_id)
            await tg_file.download_to_drive(src)

            async with gpu_lock:
                await status.edit_text("🎧 Transcribing and detecting speakers…")
                await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
                result = await asyncio.to_thread(run, src)

        if not result.turns:
            await status.edit_text("I couldn't hear any speech in that recording.")
            return

        # Role priority: caption "A, B" > /roles for this chat > Claude guess > "Speaker N".
        # A caption without commas is a description of the call, used as context for Claude.
        caption = (update.message.caption or "").strip()
        caption_roles = parse_roles(caption) if "," in caption else None
        roles = caption_roles or context.chat_data.get("roles")
        roles_source = "your names"
        if not roles and AUTO_ROLES_AVAILABLE and context.chat_data.get("auto", True):
            await status.edit_text("🧠 Working out who is who…")
            roles = await asyncio.to_thread(guess_roles, result.turns, caption or None)
            roles_source = "guessed by Claude"
        if not roles:
            roles_source = "numbered by first appearance"

        n_speakers = len({t.speaker for t in result.turns})
        header = (
            f"✅ {fmt_clock(result.duration)} · {n_speakers} speaker(s) · roles {roles_source}\n\n"
        )
        text = render("txt", result.turns, roles)
        fmt = context.chat_data.get("format", "txt")

        await status.edit_text(header.strip())
        if len(text) <= INLINE_TEXT_LIMIT:
            for chunk in _chunks(text):
                await update.message.reply_text(chunk)

        if len(text) > INLINE_TEXT_LIMIT or fmt != "txt":
            body = text if fmt == "txt" else render(fmt, result.turns, roles, segments=result.segments)
            filename = f"{Path(name).stem}_transcript.{fmt}"
            await update.message.reply_document(document=body.encode("utf-8"), filename=filename)
    except Exception as exc:  # noqa: BLE001 - report any failure back to the user
        log.exception("Processing failed")
        await status.edit_text(f"❌ Something went wrong: {type(exc).__name__}: {exc}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its request logs include the bot token

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN (from @BotFather) before starting the bot.")
    if not ALLOWED_USER_IDS:
        log.warning("ALLOWED_USER_IDS is not set: anyone who finds the bot can use it.")

    log.info("Loading models (first run downloads ~2.5 GB)…")
    load_diarizer()
    load_asr()
    log.info("Models ready.")

    app = Application.builder().token(token).concurrent_updates(True).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("roles", roles_cmd))
    app.add_handler(CommandHandler("auto", auto_cmd))
    app.add_handler(CommandHandler("format", format_cmd))
    media_filter = (
        filters.VOICE | filters.AUDIO | filters.VIDEO | filters.VIDEO_NOTE | filters.Document.AUDIO | filters.Document.VIDEO
    )
    app.add_handler(MessageHandler(media_filter, handle_media))
    log.info("Bot is running. Send it a voice message!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
