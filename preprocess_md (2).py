"""
preprocess_md.py
Place in the ROOT of the PageIndex repo (same folder as run_pageindex.py,
query_pageindex.py, and your litellm.py shim).

WHAT THIS DOES:
Takes a flat or poorly structured markdown file (no proper # headings)
and uses your LLM (via your existing litellm.py shim → CS_SLMT_Call)
to detect section boundaries and inject proper # / ## / ### headings.
The output is a structured .md file ready for run_pageindex.py.

PIPELINE:
  flat_document.md
        ↓
  preprocess_md.py        ← this script
        ↓
  flat_document_structured.md   ← proper headings added
        ↓
  run_pageindex.py --md_path flat_document_structured.md ...
        ↓
  *_structure.json
        ↓
  query_pageindex.py

WHY NO CHANGES TO litellm.py:
Your shim already handles any prompt routed through llm_completion().
This script imports the same llm_completion from pageindex.utils, so
every call goes through CS_SLMT_Call → your model exactly like indexing.

Usage:
  python preprocess_md.py --input_path "indian_culture_test.md" --model gemini/gemini-2.5-flash
  python preprocess_md.py --input_path "C:/full/path/to/doc.md" --model gemini/gemini-2.5-flash --chunk_size 150
"""

import argparse
import sys
from pathlib import Path

# Repo root must be first on sys.path so litellm.py shim is picked up,
# exactly the same as run_pageindex.py and query_pageindex.py do.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pageindex.utils import llm_completion


# ─────────────────────────────────────────────────────────────────────────────
# How many lines to send to the LLM per chunk.
# Keep this under ~150 lines so the prompt stays well within context limits.
# Overlap (OVERLAP_LINES) carries the last few lines of the previous chunk
# into the next one so the LLM has continuity at boundaries.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_LINES = 120
OVERLAP_LINES = 10

STRUCTURE_PROMPT = """\
You are a document structuring assistant.
Below is a section of a document in plain text or poorly formatted markdown.
It may have NO headings, or headings that use dashes/underlines/bold instead of # symbols.

Your job is to return the SAME text with proper markdown headings added:
- Use # for top-level sections
- Use ## for subsections
- Use ### for sub-subsections

STRICT RULES:
1. Do NOT change, remove, or rewrite any of the original content.
2. Only INSERT heading lines (lines starting with #) where section boundaries exist.
3. If a line is already a proper # heading, leave it exactly as-is.
4. If a line uses bold (**Title**) or underline (===, ---) as a heading style,
   replace it with the equivalent # heading and remove the underline line if any.
5. Return ONLY the processed text. No explanation, no preamble, no code fences.

Document section:
{chunk}
"""


def split_into_chunks(lines: list, chunk_size: int, overlap: int) -> list:
    """Split lines into overlapping chunks for processing."""
    chunks = []
    i = 0
    while i < len(lines):
        chunk = lines[i: i + chunk_size]
        chunks.append((i, chunk))
        i += chunk_size - overlap
    return chunks


def merge_chunks(chunks_output: list, original_line_count: int, chunk_size: int, overlap: int) -> list:
    """
    Merge processed chunks back into a single line list.
    For overlapping regions, prefer the version from the LATER chunk
    (it had more context about what comes next).
    Each entry in chunks_output is (start_line_index, processed_lines).
    """
    result = {}
    for chunk_idx, (start_idx, processed_lines) in enumerate(chunks_output):
        for offset, line in enumerate(processed_lines):
            abs_idx = start_idx + offset
            # Always overwrite with the later chunk's version in overlapping zones.
            result[abs_idx] = line

    # Build final list in order, filling any gaps with empty strings.
    max_idx = max(result.keys()) if result else 0
    return [result.get(i, "") for i in range(max_idx + 1)]


def process_chunk(chunk_lines: list, model: str) -> list:
    """Send one chunk to the LLM and return the structured lines."""
    chunk_text = "\n".join(chunk_lines)
    prompt = STRUCTURE_PROMPT.format(chunk=chunk_text)

    response = llm_completion(model=model, prompt=prompt).strip()

    # Strip any accidental code fences the model may add despite instructions.
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove opening fence
        lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        response = "\n".join(lines)

    return response.split("\n")


def preprocess_md(input_path: str, output_path: str, model: str,
                  chunk_size: int = DEFAULT_CHUNK_LINES, verbose: bool = True) -> str:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_text = input_file.read_text(encoding="utf-8")
    all_lines = raw_text.split("\n")
    total_lines = len(all_lines)

    if verbose:
        print(f"\n[preprocess_md] Input : {input_path}")
        print(f"[preprocess_md] Output: {output_path}")
        print(f"[preprocess_md] Model : {model}")
        print(f"[preprocess_md] Total lines: {total_lines}")
        print(f"[preprocess_md] Chunk size : {chunk_size} lines with {OVERLAP_LINES} line overlap")

    # ── Quick check: does the file already have proper headings? ──────────
    heading_lines = [l for l in all_lines if l.startswith("#")]
    heading_ratio = len(heading_lines) / max(total_lines, 1)
    if heading_ratio >= 0.02:   # 2%+ of lines are headings → probably fine already
        if verbose:
            print(f"\n[preprocess_md] File already has {len(heading_lines)} heading lines "
                  f"({heading_ratio:.1%} of content).")
            print("[preprocess_md] Skipping restructure — copying as-is.")
        Path(output_path).write_text(raw_text, encoding="utf-8")
        return output_path

    chunks = split_into_chunks(all_lines, chunk_size, OVERLAP_LINES)
    if verbose:
        print(f"[preprocess_md] Processing {len(chunks)} chunk(s)...\n")

    chunks_output = []
    for i, (start_idx, chunk_lines) in enumerate(chunks):
        if verbose:
            end_idx = min(start_idx + len(chunk_lines), total_lines)
            print(f"  Chunk {i + 1}/{len(chunks)}  (lines {start_idx + 1}–{end_idx})", end="  ", flush=True)

        processed = process_chunk(chunk_lines, model)
        chunks_output.append((start_idx, processed))

        if verbose:
            added = sum(1 for l in processed if l.startswith("#"))
            print(f"→ {added} heading(s) added/preserved")

    merged = merge_chunks(chunks_output, total_lines, chunk_size, OVERLAP_LINES)
    structured_text = "\n".join(merged)

    Path(output_path).write_text(structured_text, encoding="utf-8")

    if verbose:
        total_headings = sum(1 for l in merged if l.startswith("#"))
        print(f"\n[preprocess_md] Done. {total_headings} total heading lines in output.")
        print(f"[preprocess_md] Saved → {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add proper markdown headings to a flat/unstructured .md file before PageIndex indexing."
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to your input .md file (flat or poorly structured)"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Where to save the structured output. Defaults to <input_name>_structured.md in the same folder."
    )
    parser.add_argument(
        "--model", type=str, default="gemini/gemini-2.5-flash",
        help="Model name passed through to your litellm.py shim"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=DEFAULT_CHUNK_LINES,
        help=f"Lines per chunk sent to LLM (default: {DEFAULT_CHUNK_LINES})"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress step-by-step output"
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = args.output_path or str(
        input_path.parent / (input_path.stem + "_structured" + input_path.suffix)
    )

    result_path = preprocess_md(
        input_path=str(input_path),
        output_path=output_path,
        model=args.model,
        chunk_size=args.chunk_size,
        verbose=not args.quiet,
    )

    print(f"\nNow run PageIndex on the structured file:")
    print(f'  python run_pageindex.py --md_path "{result_path}" --model {args.model} ...')
