"""Copy the frozen SOH primary-benchmark predictions and cell metrics to reruns/primary_benchmark/,
the input location of reruns/scaling_bootstrap/scaling_bootstrap.py (Supplementary Table S12a).

Run tools/restore_data.py first: the frozen predictions are stored as lossless gzip parts.
The published bootstrap used the predictions of the 2026-11-06 retrain check, which agree with the
frozen predictions to within 2e-14 (reruns/retrain_report.md); with the frozen inputs the published
intervals are reproduced to within 3e-15 Ah and the rank and reversal tables are identical.
Existing files are never overwritten. Usage (repo root): python tools/prepare_scaling_bootstrap_inputs.py
"""
from pathlib import Path
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "project/outputs/soh_primary_benchmark_v1"
DST = ROOT / "reruns/primary_benchmark"
FILES = ["soh_primary_predictions.csv", "soh_primary_cell_metrics.csv"]


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src, dst = SRC / name, DST / name
        if not src.is_file():
            raise SystemExit(f"missing {src}; run tools/restore_data.py first")
        if dst.exists():
            if sha256(dst) == sha256(src):
                print("identical, skipped:", dst.relative_to(ROOT))
                continue
            raise SystemExit(f"{dst} exists with different content; not overwritten")
        shutil.copyfile(src, dst)
        print("copied:", dst.relative_to(ROOT), sha256(dst))


if __name__ == "__main__":
    main()
