# python /Users/khajuri/documents/podcasts/scripts/analyzer/main.py --folder test

# main.py
import argparse
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from transcriber import remote_transcribe
from analysis import analyze_chunks, save_final_analysis_output
from utils import write_text, split_transcript
from metrics import generate_pre_processing_metrics
from xml_exporter import export_as_xml
from config import (
    HOST_AUDIO_FILENAME,
    GUEST_AUDIO_FILENAME,
    CHUNK_DURATION_MIN,
)

load_dotenv()
client = OpenAI()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)
DIR_PATH = os.getenv("DIR_PATH", "")


def run(folder: str, dry_run: bool):
    host_audio_path = f"{DIR_PATH}/{folder}/audio/{HOST_AUDIO_FILENAME}"
    guest_audio_path = f"{DIR_PATH}/{folder}/audio/{GUEST_AUDIO_FILENAME}"
    output_dir = Path(f"{DIR_PATH}/{folder}/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        generate_pre_processing_metrics(client, host_audio_path, guest_audio_path)
        return

    host_segments = remote_transcribe(client, host_audio_path, "host", output_dir)
    guest_segments = remote_transcribe(client, guest_audio_path, "guest", output_dir)
    combined_segments = sorted(host_segments + guest_segments, key=lambda x: x["start"])

    host_transcript_text = "\n".join(
        [f'{seg["start"]}: {seg["text"]}' for seg in host_segments]
    )
    guest_transcript_text = "\n".join(
        [f'{seg["start"]}: {seg["text"]}' for seg in guest_segments]
    )
    combined_transcript_text = "\n".join(
        [
            f'[{seg["speaker"]}] - {seg["start"]}: {seg["text"]}'
            for seg in combined_segments
        ]
    )

    logger.info(f"Writing transcripts to output at: {output_dir}")
    write_text(output_dir / "results/sanidhya_transcript.txt", host_transcript_text)
    write_text(output_dir / "results/guest_transcript.txt", guest_transcript_text)
    write_text(output_dir / "results/combined_transcript.txt", combined_transcript_text)

    transcript_chunks = split_transcript(combined_segments, CHUNK_DURATION_MIN)
    transcript_analysis_results = analyze_chunks(
        client, transcript_chunks, output_dir, dry_run=False
    )
    save_final_analysis_output(transcript_analysis_results, output_dir)
    logger.info(f"Transcript analysis outputs saved under {output_dir}")
    export_as_xml(
        transcript_analysis_results.get("results", []),
        output_dir / "results/markers.fcpxml",
    )
    logger.info(f"Generated markers xml under {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, required=True, help="Episode folder on root path"
    )
    parser.add_argument(
        "--dryRun", action="store_true", help="Estimate token cost before analysis"
    )
    args = parser.parse_args()
    run(args.folder, args.dryRun)
