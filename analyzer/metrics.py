# analyzer/metrics.py
from pydub.utils import mediainfo
from config import WHISPER_MODEL, API_PRICING, CHUNK_DURATION_MIN
from transcriber import remote_transcribe
from analysis import analyze_chunks
from utils import split_transcript
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_pre_processing_metrics(client, host_path: str, guest_path: str) -> float:
    def get_audio_duration_minutes(audio_path):
        info = mediainfo(audio_path)
        duration_str = info.get("duration")
        if not duration_str:
            logger.warning(f"No duration found in mediainfo for: {audio_path}")
            return 0
        return float(duration_str) / 60

    host_audio_mins = get_audio_duration_minutes(host_path)
    guest_audio_mins = get_audio_duration_minutes(guest_path)
    whisper_cost_per_min = API_PRICING[WHISPER_MODEL]

    transcribe_cost = (host_audio_mins + guest_audio_mins) * whisper_cost_per_min

    dummy_segment = [{"start": 0, "end": 0, "text": "hello world"}]
    host_chunks = split_transcript(
        dummy_segment * int(CHUNK_DURATION_MIN), CHUNK_DURATION_MIN
    )
    guest_chunks = split_transcript(
        dummy_segment * int(CHUNK_DURATION_MIN), CHUNK_DURATION_MIN
    )
    est_host = analyze_chunks(client, host_chunks, "host", Path("/dev/null"), dry_run=True)
    est_guest = analyze_chunks(client, guest_chunks, "guest", Path("/dev/null"), dry_run=True)

    gpt_tokens = est_host["total_tokens"] + est_guest["total_tokens"]
    gpt_cost = est_host["cost"] + est_guest["cost"]
    total_cost = transcribe_cost + gpt_cost

    logger.info(f"COST ESTIMATION (Dry Run)")
    logger.info(
        f"Transcription (~{host_audio_mins + guest_audio_mins:.2f} min): ${transcribe_cost:.4f}"
    )
    logger.info(f"GPT Analysis ({gpt_tokens} tokens): ${gpt_cost:.4f}")
    logger.info(f"Total Estimated Cost: ${total_cost:.4f}\n")
    return total_cost
