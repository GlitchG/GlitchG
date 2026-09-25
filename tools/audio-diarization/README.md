# Who Said What: audio → transcript with speaker roles

Send an audio file and get back a transcript that shows **who said what**:

```
[0:00:00] Manager: Olá, obrigado por ter vindo. Vamos falar sobre a campanha?
[0:00:04] Client: Sim, claro. Os resultados do Meta Ads não estão bons.
[0:00:09] Manager: Vamos olhar primeiro a atribuição no GA4...
```

It uses two NVIDIA models:

| Step | Model | What it does |
|---|---|---|
| Diarization | [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) ([blog](https://huggingface.co/blog/nvidia/nemotron-diarization)) | Works out *who spoke when*. Handles up to 8 speakers, including overlapping speech |
| ASR | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Speech to text with word timestamps. Covers 25 European languages, including RU, PT and EN |

Each word goes to the speaker who was active at that moment. Consecutive words from
the same speaker are then grouped into turns.

## Quick start

### Option 1: Google Colab (free GPU, nothing to install)
Open `colab.ipynb` in Colab and select **Runtime → T4 GPU**, then **Run all**. Open the
`gradio.live` link and upload your audio.

### Option 2: Your Mac
Works on Apple Silicon (M1–M4) and Intel Macs. There's no NVIDIA GPU on a Mac, so the models run on the CPU:
slower than Colab, but your audio never leaves your computer.

Open **Terminal** (press ⌘ Space, type `Terminal`, press Enter). Paste each command and press Enter.

**1. Download the tool** (one time):
```bash
git clone https://github.com/glitchg/glitchg.git ~/WhoSaidWhat
```
If a pop-up asks to install *command line developer tools*, click **Install**, then run the command again.
If you get `destination path ... already exists`, run `cd ~/WhoSaidWhat && git pull` instead.

**2. Install** (one time, ~10–15 min, downloads ~2.5 GB). Wait for **✅ Done!**:
```bash
bash ~/WhoSaidWhat/tools/audio-diarization/mac/Install.command
```

**3. Open the app** (do this every time). It opens in your browser. Keep the Terminal window open while
you use it; close the window to stop the app:
```bash
bash ~/WhoSaidWhat/tools/audio-diarization/mac/"Who Said What.command"
```

To run the Telegram bot from your Mac instead, use this command. It keeps the Mac awake while the bot runs:
```bash
bash ~/WhoSaidWhat/tools/audio-diarization/mac/"Start Telegram Bot.command"
```

*Shortcut:* in Finder, open `WhoSaidWhat/tools/audio-diarization/mac` and double-click these files instead.
If macOS says *"unidentified developer"*, right-click the file → **Open** → **Open**.

**Keys (optional)** go in `tools/audio-diarization/.env`, which the installer creates. Open it with
`open -e ~/WhoSaidWhat/tools/audio-diarization/.env`:
- `ANTHROPIC_API_KEY=` enables Claude auto roles
- `TELEGRAM_BOT_TOKEN=` and `ALLOWED_USER_IDS=` are needed for the bot

**Terminal command:** the installer adds a `whosaid` command (open a new Terminal window first):
```bash
whosaid ~/Downloads/call.m4a --roles "Manager,Client"
whosaid ~/Downloads/call.m4a --auto-roles -f md -o ~/Desktop/transcripts/
```

**Updating later:** `cd ~/WhoSaidWhat && git pull`.

**Apple GPU (experimental):** NeMo doesn't officially support the Mac GPU. You can try it by adding
`WHOSAID_DEVICE=mps` to `.env`. If you get errors, remove that line to go back to the CPU.

### Option 3: Linux / Windows with an NVIDIA GPU
```bash
cd tools/audio-diarization
pip install -r requirements.txt      # Python 3.12+
python app.py --inbrowser            # web UI at http://127.0.0.1:7860
```

### Option 4: Telegram bot
Forward a voice message, audio file or video to your bot and get the transcript back in the chat.

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Put the token in an environment variable. **Never** paste it into code or commit it:
   ```bash
   cp .env.example .env        # then fill in TELEGRAM_BOT_TOKEN and ALLOWED_USER_IDS
   set -a; source .env; set +a
   python bot.py
   ```
3. Send `/start` to the bot. It replies with your Telegram user id. Put that id in `ALLOWED_USER_IDS`
   so that only you (and anyone else you list) can use the bot and your GPU.

What the bot does:

| You send | You get |
|---|---|
| Voice message / audio / video / video note | A transcript with speaker roles in the chat, plus a file if the transcript is long |
| Caption with commas, e.g. `Manager, Client` | Those names used as the roles |
| Caption without commas, e.g. `sales call with a hotel owner` | That description passed to Claude as context for guessing roles |
| `/roles Manager, Client` | Default roles for this chat. `/roles` on its own clears them |
| `/speakers 4` | Tells the bot how many people talk, so one person isn't split into two. `/speakers` on its own goes back to automatic |
| `/auto` | Turns automatic role detection with Claude on or off |
| `/format md` | Also sends a `.md` / `.srt` / `.json` file |

Roles are chosen in this order: caption names, then `/roles`, then Claude's guess (only if `ANTHROPIC_API_KEY` is set), then `Speaker N`.

**Limits:** Telegram only lets bots download files up to **20 MB**. Voice messages are heavily compressed,
so 20 MB is more than an hour of voice. Call recordings (m4a, wav) are often bigger. For those, use the web UI or the CLI, or
compress the file first (e.g. `ffmpeg -i call.m4a -ac 1 -b:a 32k call.ogg`).

**Hosting the bot:**
- *To try it out:* the last cell of `colab.ipynb` runs the bot on Colab's free GPU. The bot stops when the Colab session ends.
- *Always on:* run the Docker image on any machine with an NVIDIA GPU, such as a cloud GPU VM:
  ```bash
  docker build -t who-said-what .
  docker run -d --gpus all --env-file .env -v whosaid-models:/models --restart unless-stopped who-said-what
  ```

### Command line
```bash
# Print the transcript
python -m diarize_transcribe.cli call.mp3

# Name the roles (in the order people first speak) and save to a file
python -m diarize_transcribe.cli call.m4a --roles "Manager,Client" -f md -o out/

# Batch a folder of WhatsApp/Telegram voice notes into SRT subtitles
python -m diarize_transcribe.cli voice/*.ogg -f srt -o subs/
```

## Speakers split in two?
The diarizer sometimes gives one person two labels, for example when their voice or microphone changes or
they come back after a long pause. The tool fixes this automatically. It computes a voice fingerprint for
each label ([TitaNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large)) and merges
labels that sound like the same person.

If you know how many people are on the call, tell it, and it merges the labels down to exactly that number:
- **Web app:** fill in *How many people are speaking?*
- **Terminal:** `whosaid call.m4a --speakers 4`
- **Bot:** `/speakers 4`

If you give role names (`Manager, Client, Designer, Owner`), their count is used automatically.

## Roles

The diarizer numbers speakers **in order of first appearance**, so `--roles "Interviewer,Candidate"`
gives the first person to speak the name *Interviewer*. Speakers you don't name stay as `Speaker N`.

### Automatic roles with Claude
Claude can also guess roles from what people say. For example, it can tell who asks the questions,
who is being sold to, or who is addressed as "doutora". To enable it, set `ANTHROPIC_API_KEY`, then:

```bash
python -m diarize_transcribe.cli call.mp3 --auto-roles --context "sales call with a hotel owner"
```

In the web UI, tick **Guess roles with Claude**. In the bot, it runs automatically whenever no roles were given.
Role labels come back in the language of the conversation (e.g. *Gestor* / *Cliente*). The full transcript is
sent to Claude in one request, which costs a few cents for an hour-long call. If the call fails,
the tool falls back to `Speaker N`.

## Output formats

| Format | Use it for |
|---|---|
| `txt` | Reading, or pasting into docs/ChatGPT/Claude for a summary |
| `md` | Notion or Confluence, with bold speaker names |
| `srt` | Subtitles for video (Reels, YouTube) |
| `json` | Analytics: load into BigQuery or pandas (speaker talk-time, turn counts...) |

The JSON output includes `start` / `end` for every turn, so you can compute talk-time share per speaker directly.
For example, to measure how much the sales rep talks compared to the client, load the JSON into BigQuery and run:

```sql
SELECT t.speaker, ROUND(SUM(t.`end` - t.start) / 60, 1) AS minutes
FROM transcript, UNNEST(turns) AS t
GROUP BY t.speaker
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--roles` | none | Comma-separated speaker names |
| `--speakers` | number of `--roles` | How many people talk. Extra speaker labels are merged by voice |
| `--no-merge` | off | Don't merge speaker labels that have the same voice |
| `--auto-roles` | off | Let Claude guess roles (needs `ANTHROPIC_API_KEY`) |
| `--context` | none | Short description of the recording, used by `--auto-roles` |
| `-f/--format` | `txt` | `txt`, `md`, `srt` or `json` |
| `-o/--out-dir` | print to stdout | Folder to save output files in |
| `--no-timestamps` | off | Hide `[h:mm:ss]` |
| `--max-pause` | `2.0` | Start a new paragraph after a silence of this many seconds |
| `--diar-model` | `nvidia/Nemotron-3-Diarization` | If this model fails to load, the tool falls back to `nvidia/diar_streaming_sortformer_4spk-v2.1` |
| `--asr-model` | `nvidia/parakeet-tdt-0.6b-v3` | Any NeMo ASR model that returns word timestamps |

## Notes
- Any audio or video format that ffmpeg can read works (mp3, m4a, ogg, wav, mp4...). Files are converted to 16 kHz mono internally. A portable ffmpeg is installed with the Python packages, so you don't need a system-wide install.
- Recordings longer than 20 minutes switch Parakeet to local attention, which lets it handle up to about 3 hours in one pass.
- The first run downloads the models (about 2.5 GB).

## Tests
```bash
pip install pytest anthropic python-telegram-bot && python -m pytest   # no GPU or API keys needed
```
