from config import TRANSCRIBE_LANG
from utils import write_json
from pathlib import Path
import logging
from pydub import AudioSegment, silence
import tempfile
import os

logger = logging.getLogger(__name__)

MAX_SIZE_MB = 22  # Max audio chunk size for Whisper API (safe margin)
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024


def get_speaker_id(role):
    return "H" if role == "host" else "G"


def get_audio_chunks(input_path):
    audio = AudioSegment.from_file(input_path)
    chunk_paths = []

    # Rough size per ms estimate
    bytes_per_ms = len(audio.raw_data) / len(audio)
    chunk_size_ms = int(MAX_SIZE_BYTES / bytes_per_ms)

    silence_ranges = silence.detect_silence(
        audio, min_silence_len=500, silence_thresh=-40
    )
    silence_ranges = [
        (start, stop) for start, stop in silence_ranges if stop - start >= 300
    ]

    split_points = [0]
    last_split = 0
    for start, stop in silence_ranges:
        if stop - last_split > chunk_size_ms:
            split_points.append(stop)
            last_split = stop

    split_points.append(len(audio))

    for i in range(len(split_points) - 1):
        chunk = audio[split_points[i] : split_points[i + 1]]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        chunk.export(temp_file.name, format="mp3", bitrate="128k")
        logger.info(
            f"Exported chunk {i}th duration: {len(chunk)/(1000*60)}m to: {temp_file.name}"
        )
        chunk_paths.append(temp_file.name)

    return chunk_paths


def remote_transcribe(client, audio_path: str, role: str, output_dir: Path):
    logger.info(f"Chunking and transcribing {role} audio: {audio_path}")
    speakerId = get_speaker_id(role)
    chunk_paths = get_audio_chunks(audio_path)

    all_segments = []
    global_offset = 0.0

    for i, chunk_path in enumerate(chunk_paths):
        logger.info(f"Transcribing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")
        with open(chunk_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=TRANSCRIBE_LANG,
                response_format="verbose_json",
            )
        for seg in transcript.segments:
            all_segments.append(
                {
                    "start": seg.start + global_offset,
                    "end": seg.end + global_offset,
                    "text": seg.text,
                    "speaker": speakerId,
                }
            )
        # Use chunk duration (in seconds) to update offset
        chunk_audio = AudioSegment.from_file(chunk_path)
        chunk_duration_sec = len(chunk_audio) / 1000.0
        global_offset += chunk_duration_sec
        os.remove(chunk_path)

    raw_output_path = output_dir / f"raw/whisper_full_{role}_results.json"
    write_json(raw_output_path, all_segments)
    return all_segments
