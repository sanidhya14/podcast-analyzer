# analyzer/utils.py
import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def write_json(filepath: Path, data: dict | list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f"Wrote JSON: {filepath}")


def read_json(filepath: Path):
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None
    return json.loads(filepath.read_text())


def write_text(filepath: Path, text: str):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(text, encoding="utf-8")
    logger.info(f"Wrote text: {filepath}")


def read_text(filepath: Path) -> str:
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return ""
    return filepath.read_text(encoding="utf-8")


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def chunk_to_text(chunk: List[Dict]) -> str:
    return "\n".join(
        [
            f"({seg.get('speaker')}) - [{format_timestamp(seg.get('start',0))}]: {seg.get('text','').strip()}"
            for seg in chunk
        ]
    )


def split_transcript(segments: List[Dict], chunk_duration: int) -> List[List[Dict]]:
    chunks, current_chunk, current_time = [], [], 0
    for seg in segments:
        start = seg.get("start", 0)
        if start >= current_time + chunk_duration * 60:
            chunks.append(current_chunk)
            current_chunk, current_time = [], start
        current_chunk.append(seg)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
