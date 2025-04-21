# analyzer/analysis.py
import json
import tiktoken
from config import GPT_MODEL, EXTRACTION_CATEGORIES, API_PRICING
from utils import write_text, write_json, chunk_to_text
import logging
from typing import List, Dict
from pathlib import Path
from examples import TRANSCRIPT_ANALYSIS_PROMPT_EXAMPLES

logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_cost(total_tokens: int) -> float:
    return (total_tokens / 1000) * API_PRICING[GPT_MODEL]


def build_prompt(transcript_text: str, categories: List[str]) -> str:
    cats = ", ".join([f'"{c}"' for c in categories])
    return (
        "You are a helpful assistant for a youtube video podcast post-production. "
        "You will be provided a combined transcript chunk from a podcast conversation between a host and a guest. "
        "The transcript chunk will have lines format as '(speakerId) - [startTimstamp]: text'. The Guest will have speaker id as 'G' and host as 'H'."
        "Use it to extract structured metadata.\n\n"
        "Transcript is in Roman-script Hinglish or purely Engligh.\n\n"
        "TASKS:\n"
        "1. Chapters: list segments with `start_timestamp` (HH:MM:SS) and a `chapter_title` (descriptive and creative, 5–10 words).\n"
        f"2. Clip-worthy moments: extract types {cats}, each with `type`, `timestamp`, `quote` (1–2 lines, verbatim), and `why_clipworthy`.\n\n"
        "Follow below constraints:\n"
        "1. Only return valid JSON.\n"
        "2. If forced to choose between guest and host segments, priortise guest segments.\n"
        "3. Transcript can be error prone, use common sense.\n"
        "4. Focus on clarity, quotability, and meaningful moments. Feel free to ignore irrelevant or mundane segments.\n"
        "5. Do NOT hallucinate timestamps — only use those present in the transcript.\n"
        "6. Output chapter and clip segment titles, quotes etc. all in English script (transliterate if Hindi is language).\n\n"
        f"{TRANSCRIPT_ANALYSIS_PROMPT_EXAMPLES}\n\n"
        f"Transcript:\n{transcript_text}"
    )


def analyze_chunks(
    client, chunks: List[List[Dict]], output_dir: Path, dry_run: bool = True
) -> Dict:
    all_results, total_tokens = [], 0
    if dry_run:
        return {"total_tokens": total_tokens, "cost": estimate_cost(total_tokens)}

    for idx, chunk in enumerate(chunks):
        text = chunk_to_text(chunk)
        prompt = build_prompt(text, EXTRACTION_CATEGORIES)
        logger.debug(f"Using prompt: {prompt}")
        tok = count_tokens(prompt)
        total_tokens += tok

        logger.info(f"Analyzing chunk {idx+1}/{len(chunks)} ({tok} tokens)...")
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        logger.info("Received GPT model response, storing raw result in output.")
        response_content = resp.choices[0].message.content
        response_path = output_dir / f"raw/gpt_chunk{idx+1}.json"
        write_text(response_path, response_content)

        json_data_str = response_content.strip("```json\n").strip("```")
        try:
            all_results.append(json.loads(json_data_str))
        except json.JSONDecodeError:
            logger.error(f"JSON parse error in chunk {idx+1}")

    write_json(output_dir / f"raw/gpt_full_results.json", all_results)
    return {"results": all_results}


def save_final_analysis_output(data: Dict, output_dir: Path):
    chapters, moments = [], {c: [] for c in EXTRACTION_CATEGORIES}
    for res in data.get("results", []):
        chapters.extend(res.get("chapters", []))
        for m in res.get("clip_worthy_moments", []):
            t = m.get("type")
            if t in moments:
                moments[t].append(m)
    chapters_output_path = output_dir / f"results/chapters.json"
    write_text(chapters_output_path, json.dumps(chapters, indent=2))
    for cat, items in moments.items():
        clip_output_path = output_dir / f"results/{cat}_moments.json"
        write_text(clip_output_path, json.dumps(items, indent=2))
