"""
query_pageindex.py
Place in the ROOT of the PageIndex repo (same folder as run_pageindex.py
and your litellm.py shim).

WHY THIS USES THE SAME ROUTING AS INDEXING:
PageIndex's own utils.py does `import litellm` and calls
litellm.completion(...) under the hood. Because Python resolves a bare
`import litellm` against whatever's first on sys.path, and your litellm.py
shim sits in the repo root (sys.path[0] when you run a script directly from
there), every llm_completion() call below — both the tree-search step and
the answer-synthesis step — transparently goes through your shim, through
CS_SLNT_Call, to deepseek-r1. You don't need a separate API key or client.

WHAT THIS DOES:
1. Loads a tree JSON that run_pageindex.py already produced
   (run_pageindex.py saves it as "<your_pdf_name>_structure.json").
2. TREE SEARCH — sends only the lightweight structure (titles, node_ids,
   summaries — NOT full text) to the LLM and asks which nodes are likely
   to contain the answer. This mirrors PageIndex's own retrieval pattern
   (see their docs: docs.pageindex.ai/tutorials/tree-search/llm).
3. RETRIEVE — pulls the actual text of just the selected nodes out of the
   tree (no PDF re-reading needed if you indexed with --if-add-node-text yes).
4. SYNTHESIZE — sends only that retrieved text + your query to the LLM to
   produce the final answer.
Every step is printed, so you can see the full reasoning trail.

Usage:
    python query_pageindex.py --tree_path "LEGAL_CASE_DOCUMENT_structure.json" --query "What was the settlement amount?" --model deepseek-r1:latest
"""

import argparse
import json
import sys
from pathlib import Path

# Make sure the repo root (and therefore your litellm.py shim) is first on
# sys.path, exactly like run_pageindex.py relies on.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pageindex.utils import llm_completion, extract_json


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: load the tree your run_pageindex.py command already generated
# ─────────────────────────────────────────────────────────────────────────────
def load_tree(tree_path: str):
    with open(tree_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle both a raw structure list and a dict wrapping it under "structure"
    if isinstance(data, dict) and "structure" in data:
        return data["structure"]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Strip heavy fields (full node text) before sending the tree to the LLM for
# node SELECTION — keeps the reasoning step fast and cheap. This is the same
# "structure-only" pattern PageIndex's own tree search uses.
# ─────────────────────────────────────────────────────────────────────────────
def strip_text(nodes):
    lite = []
    for node in nodes:
        entry = {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "start_index": node.get("start_index"),
            "end_index": node.get("end_index"),
        }
        if node.get("summary"):
            entry["summary"] = node["summary"]
        if node.get("nodes"):
            entry["nodes"] = strip_text(node["nodes"])
        lite.append(entry)
    return lite


# ─────────────────────────────────────────────────────────────────────────────
# Flatten the tree so any node's full content can be looked up by node_id.
# ─────────────────────────────────────────────────────────────────────────────
def flatten(nodes, out=None):
    if out is None:
        out = {}
    for node in nodes:
        out[node.get("node_id")] = node
        if node.get("nodes"):
            flatten(node["nodes"], out)
    return out


def get_node_text(node: dict) -> str:
    """
    Prefer the node's own text (present if you indexed with
    --if-add-node-text yes). Fall back to concatenating children's text,
    then finally to the summary, so retrieval still works even on nodes
    that only have a summary.
    """
    if node.get("text"):
        return node["text"]
    parts = [get_node_text(child) for child in node.get("nodes", [])]
    parts = [p for p in parts if p]
    if parts:
        return "\n\n".join(parts)
    return node.get("summary", "")


TREE_SEARCH_PROMPT = """You are given a query and the tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure:
{tree}

Reply in the following JSON format:
{{
  "thinking": <your reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}

Directly return the final JSON structure. Do not output anything else."""

SYNTHESIS_PROMPT = """You are given a query and the retrieved content of relevant document sections.
Answer the query using only the information in the retrieved content below.
If the retrieved content does not contain enough information to answer, say so clearly.

Query: {query}

Retrieved sections:
{context}

Reply in the following JSON format:
{{
  "thinking": <how you arrived at the answer from the retrieved content>,
  "answer": <your final answer to the query>
}}

Directly return the final JSON structure. Do not output anything else."""


def query_pageindex(tree_path: str, query: str, model: str, verbose: bool = True) -> dict:
    tree = load_tree(tree_path)
    lite_tree = strip_text(tree)
    flat = flatten(tree)

    # ── Step 1: tree search ────────────────────────────────────────────────
    if verbose:
        print("=" * 70)
        print("STEP 1 — Tree search (asking the LLM which nodes are relevant)")
        print("=" * 70)

    search_prompt = TREE_SEARCH_PROMPT.format(
        query=query, tree=json.dumps(lite_tree, indent=2)
    )
    raw_search_response = llm_completion(model=model, prompt=search_prompt)
    search_result = extract_json(raw_search_response)
    node_ids = search_result.get("node_list", [])

    if verbose:
        print(f"\n[LLM reasoning]\n{search_result.get('thinking', '')}")
        print(f"\n[Selected node_ids]: {node_ids}")

    # ── Step 2: retrieve the actual content of those nodes ─────────────────
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2 — Retrieving content for the selected nodes")
        print("=" * 70)

    context_parts = []
    for nid in node_ids:
        node = flat.get(nid)
        if not node:
            if verbose:
                print(f"\n[warn] node_id {nid!r} not found in tree, skipping")
            continue
        text = get_node_text(node)
        context_parts.append(
            f"### {node.get('title')} (node_id: {nid}, pages "
            f"{node.get('start_index')}-{node.get('end_index')})\n{text}"
        )
        if verbose:
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"\n[Retrieved] {node.get('title')!r} (node_id: {nid})\n{preview}")

    context = "\n\n".join(context_parts) if context_parts else "(no content retrieved)"

    # ── Step 3: synthesize the final answer ─────────────────────────────────
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3 — Synthesizing the answer from retrieved content")
        print("=" * 70)

    synth_prompt = SYNTHESIS_PROMPT.format(query=query, context=context)
    raw_synth_response = llm_completion(model=model, prompt=synth_prompt)
    synth_result = extract_json(raw_synth_response)

    if verbose:
        print(f"\n[LLM reasoning]\n{synth_result.get('thinking', '')}")
        print(f"\n[FINAL ANSWER]\n{synth_result.get('answer', '')}")

    return {
        "selected_nodes": node_ids,
        "search_reasoning": search_result.get("thinking", ""),
        "answer": synth_result.get("answer", ""),
        "answer_reasoning": synth_result.get("thinking", ""),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a PageIndex tree and trace how the answer is retrieved."
    )
    parser.add_argument("--tree_path", type=str, required=True, help="Path to the generated *_structure.json file")
    parser.add_argument("--query", type=str, required=True, help="The question to ask")
    parser.add_argument(
        "--model", type=str, default="deepseek-r1:latest",
        help="Model name passed through to your litellm.py shim (same value you used to index)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step trace, only print the final answer")
    args = parser.parse_args()

    result = query_pageindex(args.tree_path, args.query, args.model, verbose=not args.quiet)

    if args.quiet:
        print(result["answer"])
