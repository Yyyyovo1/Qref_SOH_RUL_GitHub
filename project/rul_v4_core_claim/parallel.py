"""Process-level parallel execution that does not change any computation (implementation notes 6.11).

Jobs are independent pure functions of their arguments; results are returned in submission order, so the merged
outputs are identical whatever the number of workers. Worker processes are started with "spawn" (as on Windows).
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def run_jobs(func, arg_list: list[tuple], workers: int = 1, progress=None, every: int = 50) -> list:
    """`func(*args)` for every tuple in `arg_list`, in order. workers <= 1 runs in this process."""
    if workers <= 1 or len(arg_list) <= 1:
        out = []
        for i, a in enumerate(arg_list, 1):
            out.append(func(*a))
            if progress and (i % every == 0 or i == len(arg_list)):
                progress(i, len(arg_list))
        return out
    with ProcessPoolExecutor(max_workers=min(workers, len(arg_list)), mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(func, *a) for a in arg_list]
        out = []
        for i, f in enumerate(futures, 1):
            out.append(f.result())
            if progress and (i % every == 0 or i == len(arg_list)):
                progress(i, len(arg_list))
        return out
