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

### Option 2: Local machine (an NVIDIA GPU is recommended; CPU works but is slow)
```bash
cd tools/audio-diarization
pip install -r requirements.txt      # also install ffmpeg: brew/apt install ffmpeg
python app.py                        # web UI at http://127.0.0.1:7860
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

## Roles

The diarizer numbers speakers **in order of first appearance**, so `--roles "Interviewer,Candidate"`
gives the first person to speak the name *Interviewer*. Speakers you don't name stay as `Speaker N`.

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
| `-f/--format` | `txt` | `txt`, `md`, `srt` or `json` |
| `-o/--out-dir` | print to stdout | Folder to save output files in |
| `--no-timestamps` | off | Hide `[h:mm:ss]` |
| `--max-pause` | `2.0` | Start a new paragraph after a silence of this many seconds |
| `--diar-model` | `nvidia/Nemotron-3-Diarization` | If this model fails to load, the tool falls back to `nvidia/diar_streaming_sortformer_4spk-v2.1` |
| `--asr-model` | `nvidia/parakeet-tdt-0.6b-v3` | Any NeMo ASR model that returns word timestamps |

## Notes
- Any audio or video format that ffmpeg can read works (mp3, m4a, ogg, wav, mp4...). Files are converted to 16 kHz mono internally.
- Recordings longer than 20 minutes switch Parakeet to local attention, which lets it handle up to about 3 hours in one pass.
- The first run downloads the models (about 2.5 GB).

## Tests
```bash
pip install pytest && python -m pytest    # alignment + formatting logic, no GPU needed
```
