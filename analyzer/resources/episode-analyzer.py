import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from faster_whisper import WhisperModel
import openai
import tiktoken

# === CONFIGURATION ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Whisper model config
WHISPER_MODEL_SIZE = "large-v3"
TRANSCRIBE_LANG = "hi"  # for Hinglish in Roman script
CHUNK_DURATION_MIN = 10  # minutes per chunk

# File input/output paths
HOST_AUDIO_PATH_TEMPLATE = "./{}/audio/sanidhya.wav"
GUEST_AUDIO_PATH_TEMPLATE = "./{}/audio/guest.wav"

# Metadata categories to extract
EXTRACTION_CATEGORIES = ["emotional", "intellectual", "vulnerable", "high-energy"]

# OpenAI model config
GPT_MODEL = "gpt-3.5-turbo"
TOKEN_COST_PER_1K = 0.001  # USD for gpt-3.5-turbo

# === UTILITIES ===
def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def split_transcript(segments: List[Dict], chunk_duration: int) -> List[List[Dict]]:
    chunks, current_chunk, current_time = [], [], 0
    for seg in segments:
        start = seg['start']
        if start >= current_time + chunk_duration * 60:
            chunks.append(current_chunk)
            current_chunk = []
            current_time = start
        current_chunk.append(seg)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def transcribe(audio_path: str, model: WhisperModel) -> List[Dict]:
    print(f"Transcribing: {audio_path}")
    segments, _ = model.transcribe(audio_path, language=TRANSCRIBE_LANG)
    return [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        } for seg in segments
    ]

def remote_transcribe(audio_path: str) -> List[Dict]:
    print(f"Remote transcribing with OpenAI: {audio_path}")
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file, language=TRANSCRIBE_LANG)
    # Simulate structure similar to faster-whisper output
    return [{"start": 0, "end": 0, "text": transcript["text"]}]

def chunk_to_text(chunk: List[Dict]) -> str:
    return "\n".join([
        f"[{format_timestamp(seg['start'])}] {seg['text']}"
        for seg in chunk
    ])

def count_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(total_tokens: int) -> float:
    return (total_tokens / 1000) * TOKEN_COST_PER_1K

def build_prompt(transcript_text: str, categories: List[str]) -> str:
    categories_str = ", ".join([f'"{cat}"' for cat in categories])
    return f'''
You are a helpful assistant for podcast post-production. You will be provided a transcript chunk from a podcast conversation between a host and a guest. Use it to extract structured metadata.

Transcript is in Hinglish (a mix of Hindi and English), presented in Roman script.

TASKS:
1. **Chapters**
   - Identify natural chapter segments in this chunk
   - Provide a list of chapters with:
     - `start_timestamp` (format: HH:MM:SS)
     - `chapter_title` (descriptive and creative, 5–10 words)

2. **Clip-worthy Moments**
   - Extract moments by the following types: {categories_str}.
   - For each moment, include:
     - `type` (e.g., one of {categories_str})
     - `timestamp`
     - `quote` (1–2 lines, verbatim)
     - `why_clipworthy` (brief note on impact, relatability, or virality)

Only return valid JSON.

Transcript:
{transcript_text}
'''

def analyze_chunks(chunks: List[List[Dict]], role: str, dry_run: bool = True) -> Dict:
    all_results = []
    total_tokens = 0

    for idx, chunk in enumerate(chunks):
        text = chunk_to_text(chunk)
        prompt = build_prompt(text, EXTRACTION_CATEGORIES)
        tokens = count_tokens(prompt)
        total_tokens += tokens

        if dry_run:
            continue

        print(f"\nAnalyzing {role} chunk {idx + 1}/{len(chunks)} ({tokens} tokens)...")
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        try:
            result = json.loads(response.choices[0].message['content'])
            all_results.append(result)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse response JSON.")

    if dry_run:
        print(f"\n🧮 Total estimated tokens for {role}: {total_tokens}")
        print(f"💰 Estimated cost: ${estimate_cost(total_tokens):.4f}")
        return {"total_tokens": total_tokens, "cost": estimate_cost(total_tokens)}

    return {"results": all_results}

def save_output(data: Dict, role: str):
    # Save results per category
    all_chapters = []
    all_moments = {cat: [] for cat in EXTRACTION_CATEGORIES}

    for chunk_result in data['results']:
        all_chapters.extend(chunk_result.get("chapters", []))
        for moment in chunk_result.get("moments", []):
            cat = moment.get("type")
            if cat in all_moments:
                all_moments[cat].append(moment)

    with open(OUTPUT_DIR / f"{role}_chapters.json", "w") as f:
        json.dump(all_chapters, f, indent=2)

    for cat, items in all_moments.items():
        with open(OUTPUT_DIR / f"{role}_{cat}_moments.json", "w") as f:
            json.dump(items, f, indent=2)

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Episode folder name on root path')
    parser.add_argument("--dryRun", action="store_true", help="Perform dry run to estimate token cost only")
    parser.add_argument("--remote-transcribe", action="store_true", help="Use OpenAI Whisper API instead of local Whisper model")
    args = parser.parse_args()
    folder_name = args.folder
    dry_run_mode = args.dryRun
    remote_mode = args.remote_transcribe

    HOST_AUDIO_PATH = HOST_AUDIO_PATH_TEMPLATE.format(folder_name)
    GUEST_AUDIO_PATH = GUEST_AUDIO_PATH_TEMPLATE.format(folder_name)
    OUTPUT_DIR = Path(f"./{folder_name}/output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not remote_mode:
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    for role, audio_path in [("host", HOST_AUDIO_PATH), ("guest", GUEST_AUDIO_PATH)]:
        if remote_mode:
            segments = remote_transcribe(audio_path)
        else:
            segments = transcribe(audio_path, model)

        chunks = split_transcript(segments, chunk_duration=CHUNK_DURATION_MIN)

        print(f"\n--- {'DRY RUN' if dry_run_mode else 'PROCESSING'}: {role.upper()} ---")
        result = analyze_chunks(chunks, role, dry_run=dry_run_mode)

        if dry_run_mode:
            proceed = input(f"Proceed with OpenAI API call for {role}? (y/n): ").strip().lower()
            if proceed != "y":
                print(f"⏭️ Skipped {role} API calls.")
                continue
            result = analyze_chunks(chunks, role, dry_run=False)

        save_output(result, role)
        print(f"✅ Saved outputs for {role} to {OUTPUT_DIR}")
