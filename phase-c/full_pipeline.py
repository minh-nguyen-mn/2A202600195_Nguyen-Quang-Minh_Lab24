import asyncio
import time
import numpy as np

from input_guard import InputGuard


input_guard = InputGuard()


async def rag_pipeline_async(question):
    await asyncio.sleep(0.1)
    return f"Answer for: {question}"


async def refuse_response():
    return "Request blocked by guardrails."


async def audit_log(user_input, answer, timings):
    return


async def guarded_pipeline(user_input):
    timings = {}

    t0 = time.perf_counter()

    sanitized, _ = input_guard.sanitize(user_input)

    timings["L1"] = (
        time.perf_counter() - t0
    ) * 1000

    t0 = time.perf_counter()

    answer = await rag_pipeline_async(sanitized)

    timings["L2"] = (
        time.perf_counter() - t0
    ) * 1000

    t0 = time.perf_counter()

    await asyncio.sleep(0.05)

    timings["L3"] = (
        time.perf_counter() - t0
    ) * 1000

    asyncio.create_task(
        audit_log(user_input, answer, timings)
    )

    return answer, timings


async def benchmark(n=100):
    queries = [f"Query {i}" for i in range(n)]

    all_timings = []

    for q in queries:
        _, t = await guarded_pipeline(q)

        all_timings.append(t)

    for layer in ["L1", "L2", "L3"]:
        vals = [
            t[layer]
            for t in all_timings
        ]

        print(
            f"{layer}: "
            f"P50={np.percentile(vals,50):.2f}ms "
            f"P95={np.percentile(vals,95):.2f}ms "
            f"P99={np.percentile(vals,99):.2f}ms"
        )


if __name__ == "__main__":
    asyncio.run(benchmark())