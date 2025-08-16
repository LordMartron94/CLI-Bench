import contextlib
import gc
import os

import psutil
import torch


# noinspection t
@contextlib.contextmanager
def stable_timing(threads: int = 1, pin_affinity: bool = False):
    # threads
    old_env = {k: os.getenv(k) for k in ["MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"]}
    for k in old_env: os.environ[k] = str(threads)
    old_torch = torch.get_num_threads() if torch else None
    if torch: torch.set_num_threads(threads)

    # GC
    gc_was = gc.isenabled(); gc.collect(); gc.disable()

    # affinity (optional)
    proc = psutil.Process()
    old_aff = proc.cpu_affinity() if pin_affinity and hasattr(proc, "cpu_affinity") else None
    if old_aff is not None: proc.cpu_affinity(old_aff[:max(1, threads)])

    try:
        yield
    finally:
        if old_aff is not None: proc.cpu_affinity(old_aff)
        if gc_was: gc.enable()
        for k,v in old_env.items(): os.environ[k] = v if v is not None else ""
        if torch and old_torch is not None: torch.set_num_threads(old_torch)
