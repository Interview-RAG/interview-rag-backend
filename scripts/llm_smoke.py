"""Fire N requests at one tier and show which deployment served each.

    python scripts/llm_smoke.py --tier fast --n 30
    python scripts/llm_smoke.py --tier heavy --n 12
    python scripts/llm_smoke.py --tier vision --image sample.png
    python scripts/llm_smoke.py --tier agent --n 2        # LangChain path
    LLM_GEMINI_FLASH_RPM=1 python scripts/llm_smoke.py --tier heavy --n 3

Prints one line per call, then the router snapshot. Expect zero 429s in the
log: the point of the router is to stop before the provider has to.
"""
import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

import llm  # noqa: E402


async def one_chat(tier: str, i: int, image: str | None):
    content = f"Reply with the single word OK and the number {i}."
    if image:
        mime = mimetypes.guess_type(image)[0] or "image/png"
        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content = [
            {"type": "text", "text": "List any interview questions you can read in this image as a JSON object {\"qa_pairs\": [...]}."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    started = time.monotonic()
    try:
        response = await llm.chat(
            tier,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=256,
            reasoning_effort="none",
        )
        text = (response.choices[0].message.content or "").strip().replace("\n", " ")
        served = llm.snapshot()["recent"][-1]["dep"]
        print(f"[{i:02d}] ok   {served:<14} {int((time.monotonic() - started) * 1000):>5}ms  {text[:60]}")
    except Exception as e:
        print(f"[{i:02d}] FAIL {type(e).__name__}: {str(e)[:120]}")


async def one_agent(i: int):
    from langchain_core.messages import HumanMessage, SystemMessage
    import rag_agent

    started = time.monotonic()
    try:
        ai = await rag_agent.invoke_agent_llm([
            SystemMessage(content="You are a terse assistant. Use tools only if needed."),
            HumanMessage(content=f"Say OK {i}. Do not call any tool."),
        ])
        served = llm.snapshot()["recent"][-1]["dep"]
        print(f"[{i:02d}] ok   {served:<14} {int((time.monotonic() - started) * 1000):>5}ms  {str(ai.content)[:60]!r} tools={len(ai.tool_calls)}")
    except Exception as e:
        print(f"[{i:02d}] FAIL {type(e).__name__}: {str(e)[:160]}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="fast")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--image", default=None)
    ap.add_argument("--sequential", action="store_true", help="one at a time instead of all at once")
    args = ap.parse_args()

    print(f"tier={args.tier} deployments={[d.name for d in llm.deployments_for(args.tier)]} "
          f"enabled={[d.name for d in llm.deployments_for(args.tier) if d.enabled()]}")

    if args.tier == "agent":
        jobs = [one_agent(i) for i in range(1, args.n + 1)]
    else:
        jobs = [one_chat(args.tier, i, args.image) for i in range(1, args.n + 1)]

    if args.sequential:
        for job in jobs:
            await job
    else:
        await asyncio.gather(*jobs)

    snap = llm.snapshot()
    print("\n--- deployments ---")
    for name, s in snap["deployments"].items():
        print(f"{name:<14} {s['provider']:<7} {s['model']:<28} enabled={s['enabled']!s:<5} "
              f"rpm={s['rpm']['used']}/{s['rpm']['limit']} tpm={s['tpm']['used']}/{s['tpm']['limit']} "
              f"rpd={s['rpd']['used']}/{s['rpd']['limit']} pools={s['pools']} "
              f"served={s['served']} failed={s['failed']} skipped={s['skipped']} "
              f"cool={s['cooldown_s']}s {s['cooldown_reason']} {s['last_error'][:80]}")
    served_by = {}
    for r in snap["recent"]:
        if r["status"] == "ok":
            served_by[r["dep"]] = served_by.get(r["dep"], 0) + 1
    print(f"\nserved by: {json.dumps(served_by)}")


if __name__ == "__main__":
    asyncio.run(main())
