import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# === USER-CONFIGURABLE PARAMETERS ===
TRANSCRIBE_LANG = "hi"  # for Hinglish in Roman script
CHUNK_DURATION_MIN = 10  # minutes per chunk
HOST_AUDIO_FILENAME = "sanidhya.wav"
GUEST_AUDIO_FILENAME = "guest.wav"
EXTRACTION_CATEGORIES = ["emotional", "intellectual", "vulnerable", "high-energy"]
GPT_MODEL = "gpt-3.5-turbo"
WHISPER_MODEL = "whisper"
API_PRICING = {
    GPT_MODEL : 0.001 # USD per 1000 tokens,
    WHISPER_MODEL:  0.006 # USD per minute audio
}  

# === STATIC PATH TEMPLATES ===
HOST_AUDIO_PATH_TEMPLATE = "./{}/audio/" + HOST_AUDIO_FILENAME
GUEST_AUDIO_PATH_TEMPLATE = "./{}/audio/" + GUEST_AUDIO_FILENAME

load_dotenv()
client = OpenAI()

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def split_transcript(segments: List[Dict], chunk_duration: int) -> List[List[Dict]]:
    chunks, current_chunk, current_time = [], [], 0
    for seg in segments:
        start = seg.get('start', 0)
        if start >= current_time + chunk_duration * 60:
            chunks.append(current_chunk)
            current_chunk, current_time = [], start
        current_chunk.append(seg)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def remote_transcribe(audio_path: str) -> List[Dict]:
    print(f"Remote transcribing with OpenAI Whisper API: {audio_path}")
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=TRANSCRIBE_LANG
        )
    return [{"start": 0, "end": 0, "text": transcript.text}]

def chunk_to_text(chunk: List[Dict]) -> str:
    return "\n".join([
        f"[{format_timestamp(seg.get('start',0))}] {seg.get('text','').strip()}"
        for seg in chunk
    ])

def count_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(total_tokens: int) -> float:
    return (total_tokens / 1000) * API_PRICING[GPT_MODEL]

def build_prompt(transcript_text: str, categories: List[str]) -> str:
    cats = ", ".join([f'"{c}"' for c in categories])
    return (
        "You are a helpful assistant for podcast post-production. "
        "You will be provided a transcript chunk from a podcast conversation between a host and a guest. "
        "Use it to extract structured metadata.\n\n"
        "Transcript is in Roman-script Hinglish.\n\n"
        "TASKS:\n"
        "1. Chapters: list segments with start_timestamp (HH:MM:SS) and a 5–10 word title.\n"
        f"2. Clip-worthy moments: extract types {cats}, each with type, timestamp, quote (1–2 lines), and why_clipworthy.\n\n"
        "Only return valid JSON.\n\n"
        f"Transcript:\n{transcript_text}"
    )

def analyze_chunks(chunks: List[List[Dict]], role: str, dry_run: bool = True) -> Dict:
    all_results, total_tokens = [], 0
    for idx, chunk in enumerate(chunks):
        text = chunk_to_text(chunk)
        prompt = build_prompt(text, EXTRACTION_CATEGORIES)
        tok = count_tokens(prompt)
        total_tokens += tok
        if not dry_run:
            print(f"Analyzing {role} chunk {idx+1}/{len(chunks)} ({tok} tokens)...")
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            try:
                all_results.append(json.loads(resp.choices[0].message.content))
            except json.JSONDecodeError:
                print(f"⚠️ JSON parse error in chunk {idx+1}")
    if dry_run:
        return {"total_tokens": total_tokens, "cost": estimate_cost(total_tokens)}
    return {"results": all_results}

def save_output(data: Dict, role: str, output_dir: Path):
    chapters, moments = [], {c: [] for c in EXTRACTION_CATEGORIES}
    for res in data.get('results', []):
        chapters.extend(res.get('chapters', []))
        for m in res.get('moments', []):
            t = m.get('type')
            if t in moments:
                moments[t].append(m)
    (output_dir / f"{role}_chapters.json").write_text(json.dumps(chapters, indent=2))
    for cat, items in moments.items():
        (output_dir / f"{role}_{cat}_moments.json").write_text(json.dumps(items, indent=2))

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Episode folder on root path")
    parser.add_argument("--dryRun", action="store_true", help="Estimate token cost before analysis")
    args = parser.parse_args()

    folder = args.folder
    dry_run = args.dryRun

    HOST_AUDIO_PATH = HOST_AUDIO_PATH_TEMPLATE.format(folder)
    GUEST_AUDIO_PATH = GUEST_AUDIO_PATH_TEMPLATE.format(folder)
    OUTPUT_DIR = Path(f"./{folder}/output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def generate_pre_processing_metrics():

        def get_audio_duration_minutes(audio_path):
            info = mediainfo(audio_path)
            duration_sec = float(info['duration'])
            return duration_sec / 60
    
        host_audio_mins = get_audio_duration_minutes(HOST_AUDIO_PATH)
        guest_audio_mins = get_audio_duration_minutes(GUEST_AUDIO_PATH)
        whisper_cost_per_min = API_PRICING[WHISPER_MODEL]

        transcribe_cost = (host_audio_mins + guest_audio_mins) * whisper_cost_per_min

        # Dummy segments just to get GPT token estimate (without hitting APIs)
        dummy_segment = [{"start": 0, "end": 0, "text": "hello world"}]
        host_chunks = split_transcript(dummy_segment * int(CHUNK_DURATION_MIN), CHUNK_DURATION_MIN)
        guest_chunks = split_transcript(dummy_segment * int(CHUNK_DURATION_MIN), CHUNK_DURATION_MIN)
        est_host = analyze_chunks(host_chunks, 'host', dry_run=True)
        est_guest = analyze_chunks(guest_chunks, 'guest', dry_run=True)

        gpt_tokens = est_host['total_tokens'] + est_guest['total_tokens']
        gpt_cost = est_host['cost'] + est_guest['cost']
        total_cost = transcribe_cost + gpt_cost

        print(f"\n📊 COST ESTIMATION (Dry Run)")
        print(f"🎙️  Transcription (~{host_audio_mins + guest_audio_mins:.2f} min): ${transcribe_cost:.4f}")
        print(f"🧠  GPT Analysis ({gpt_tokens} tokens): ${gpt_cost:.4f}")
        print(f"💰  Total Estimated Cost: ${total_cost:.4f}\n")

     if dry_run:
        return
        
    
        
