# GPU image for the Telegram bot (or the web UI). NeMo Speech container already ships
# CUDA, PyTorch and NeMo, so only the small extras are installed here.
FROM nvcr.io/nvidia/nemo-speech:26.07.00

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
# Newer NeMo than the container ships: Nemotron 3 Diarization needs RoPE attention support.
RUN pip install --no-cache-dir "nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@cf724ac337d1ebc7d0dda1e23fb80916f52927a5" "python-telegram-bot>=21" "anthropic>=1.8" "gradio>=4.44" imageio-ffmpeg

WORKDIR /app
COPY diarize_transcribe/ diarize_transcribe/
COPY bot.py app.py ./

# Model downloads are cached here; mount a volume to keep them between restarts.
ENV HF_HOME=/models
VOLUME /models

CMD ["python", "bot.py"]
