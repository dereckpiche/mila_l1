"""
Measures how long a model thinks on L1's training prompts, uncapped.

    python scripts/probe_lengths.py Qwen/Qwen3.5-4B [data.parquet]

Environment: THINK (1: thinking on), THINK_CLOSE (the tag that closes the
thinking: </think> for Qwen3.5, <channel|> for Gemma 4), TEMP, TOP_P, TOP_K,
PRESENCE.
"""

import os
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from math_reward import math_reward_fn

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/exact/train.parquet"
NUM_PROMPTS = 64
MAX_TOKENS = 16384
THINK = os.environ.get("THINK", "1") == "1"
THINK_CLOSE = os.environ.get("THINK_CLOSE", "</think>")
SAMPLING = dict(
    temperature=float(os.environ.get("TEMP", 0.6)),
    top_p=float(os.environ.get("TOP_P", 1.0)),
    top_k=int(os.environ.get("TOP_K", -1)),
    presence_penalty=float(os.environ.get("PRESENCE", 0.0)),
)


def main():
    """Sample uncapped responses and print length and </think> statistics."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = pd.read_parquet(DATA).sample(NUM_PROMPTS, random_state=0)
    prompts = [
        tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            enable_thinking=THINK,
            tokenize=False,
        )
        for messages in rows["prompt"]
    ]
    print("PROMPT TAIL", repr(prompts[0][-300:]))
    llm = LLM(MODEL, max_model_len=MAX_TOKENS + 1024, gpu_memory_utilization=0.85)
    print("SAMPLING", SAMPLING)
    params = SamplingParams(
        max_tokens=MAX_TOKENS, skip_special_tokens=False, **SAMPLING
    )
    outputs = llm.generate(prompts, params)
    lengths = np.array([len(o.outputs[0].token_ids) for o in outputs])
    texts = [
        o.outputs[0].text.replace(THINK_CLOSE, "</think>") for o in outputs
    ]
    think_ends = np.array(
        [
            len(tokenizer(t.split("</think>")[0])["input_ids"]) if "</think>" in t else -1
            for t in texts
        ]
    )
    closed = think_ends >= 0
    print("RESPONSE LENGTH percentiles 10/25/50/75/90:", np.percentile(lengths, [10, 25, 50, 75, 90]).astype(int))
    print("CLOSED </think>:", closed.mean())
    print("THINK LENGTH percentiles (closed only):", np.percentile(think_ends[closed], [10, 25, 50, 75, 90]).astype(int) if closed.any() else None)
    for cap in (4096, 8192, 12288):
        print(f"FINISHED within {cap}:", (lengths < cap).mean(), " CLOSED think within", cap, ":", ((think_ends >= 0) & (think_ends < cap)).mean())
    print("BOXED:", np.mean(["\\boxed" in t for t in texts]))
    correct = [
        math_reward_fn(t, row["ground_truth"], ignore_think_token=not THINK) > 0.5
        for t, row in zip(texts, rows["reward_model"])
    ]
    print("THINK", THINK, "CORRECT:", np.mean(correct))
    print("SAMPLE HEAD", repr(texts[0][:400]))
    print("SAMPLE TAIL", repr(texts[0][-400:]))


if __name__ == "__main__":
    main()
