#!/usr/bin/env python3
"""
csv_to_h5.py — Convert Scope B CSVs to HDF5 for memory-safe training.

Why this exists:
  Flexynesis CSV DataImporter blows RAM (60+ GB peak with copies) on the
  Scope B 14 GB train CSV → GUI freeze + force-shutdown. HDF5 with
  lazy per-sample loading sidesteps the whole problem.

Inputs (must exist):
  processed_scaled_411k_tissue_B/train/gex.csv         (14 GB, ~118k samples)
  processed_scaled_411k_tissue_B/train/clin.csv
  processed_scaled_411k_tissue_B/test/gex.csv          (3.3 GB, ~28k samples)
  processed_scaled_411k_tissue_B/test/clin.csv

Outputs:
  processed_scaled_411k_tissue_B_h5/train/gex.h5       (~8 GB, samples × genes)
  processed_scaled_411k_tissue_B_h5/train/clin.csv     (copied)
  processed_scaled_411k_tissue_B_h5/test/gex.h5        (~2 GB)
  processed_scaled_411k_tissue_B_h5/test/clin.csv      (copied)

HDF5 layout per gex.h5:
  /expression    (n_samples, n_genes) float32, chunks=(1, n_genes)  ← fast row reads
  /sample_ids    (n_samples,) bytes
  /gene_symbols  (n_genes,) bytes
  attrs: created_by, source_csv, normalization, orientation

Memory profile:
  Read phase:      ~8 GB (pre-allocated array fills as chunks parsed)
  Transpose phase: ~16 GB (temporary 2× during transpose copy)
  Write phase:     ~8 GB (writes to disk, then frees)
  PEAK:            ~16 GB ← well within 58 GB free

Runtime: ~6-8 min total (NVMe sequential, no compression overhead)
"""
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT    = Path("/home/amit/Desktop/projects/flexynesis")
SRC_DIR = ROOT / "processed_scaled_411k_tissue_B"
DST_DIR = ROOT / "processed_scaled_411k_tissue_B_h5"

GENES_PER_CHUNK = 500   # pandas chunksize


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def convert_split(split):
    src_gex  = SRC_DIR / split / "gex.csv"
    src_clin = SRC_DIR / split / "clin.csv"
    dst_gex  = DST_DIR / split / "gex.h5"
    dst_clin = DST_DIR / split / "clin.csv"

    DST_DIR.joinpath(split).mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"[{split}] CSV → HDF5")
    log("=" * 70)

    # ---------- Copy clin.csv ----------
    shutil.copy(src_clin, dst_clin)
    log(f"  Copied clin.csv → {dst_clin}")

    # ---------- Pass 1: scan structure (headers + gene names) ----------
    log("  [1/4] Scanning CSV structure...")
    t = time.time()

    # Get sample IDs from header (one-row read, near-instant)
    header_df = pd.read_csv(src_gex, nrows=0, index_col=0)
    sample_ids = header_df.columns.tolist()
    n_samples = len(sample_ids)

    # Get gene symbols (first column only — fast)
    gene_col = pd.read_csv(src_gex, usecols=[0])
    gene_symbols = gene_col.iloc[:, 0].tolist()
    n_genes = len(gene_symbols)
    del gene_col

    log(f"      Samples: {n_samples:,}  Genes: {n_genes:,}  "
        f"(scan {time.time()-t:.1f}s)")
    log(f"      Output size: {n_samples*n_genes*4/1e9:.2f} GB (float32, no compression)")

    # ---------- Pass 2: stream CSV into pre-allocated array ----------
    log("  [2/4] Streaming CSV into RAM (pandas chunked, float32)...")
    t = time.time()

    # Pre-allocate target: (n_genes, n_samples) — matches CSV orientation
    arr = np.empty((n_genes, n_samples), dtype=np.float32)

    chunks = pd.read_csv(src_gex, index_col=0, chunksize=GENES_PER_CHUNK)
    gene_idx = 0
    for chunk_i, chunk in enumerate(chunks):
        chunk_arr = chunk.values.astype(np.float32, copy=False)  # (n, n_samples)
        n = chunk_arr.shape[0]
        arr[gene_idx:gene_idx + n] = chunk_arr
        gene_idx += n
        del chunk_arr

        if (chunk_i + 1) % 4 == 0:
            elapsed = time.time() - t
            rate = gene_idx / elapsed
            eta_min = (n_genes - gene_idx) / rate / 60
            log(f"      {gene_idx:>6,}/{n_genes:,} genes  "
                f"({rate:.0f}/s, ETA {eta_min:.1f} min, elapsed {elapsed/60:.1f} min)")

    read_min = (time.time() - t) / 60
    log(f"      Read done: {arr.nbytes / 1e9:.2f} GB in RAM  ({read_min:.1f} min)")

    # ---------- Pass 3: transpose to (n_samples, n_genes) ----------
    log("  [3/4] Transposing (n_genes, n_samples) → (n_samples, n_genes)...")
    t = time.time()
    # ascontiguousarray forces a real copy in C-order — needed for HDF5 write
    arr_T = np.ascontiguousarray(arr.T)
    del arr  # free 8 GB
    log(f"      Transpose done ({time.time()-t:.1f}s)  "
        f"Now: {arr_T.nbytes / 1e9:.2f} GB")

    # ---------- Pass 4: write HDF5 ----------
    log("  [4/4] Writing HDF5...")
    t = time.time()
    with h5py.File(dst_gex, "w") as h5f:
        # chunks=(1, n_genes) → one chunk = one sample row = fast single-sample reads
        h5f.create_dataset(
            "expression",
            data=arr_T,
            chunks=(1, n_genes),
            # No compression — speed > space (NVMe has 464 GB free)
        )
        h5f.create_dataset("sample_ids",
                           data=np.array(sample_ids, dtype="S"))
        h5f.create_dataset("gene_symbols",
                           data=np.array(gene_symbols, dtype="S"))
        h5f.attrs["created_by"]    = "csv_to_h5.py"
        h5f.attrs["source_csv"]    = str(src_gex)
        h5f.attrs["normalization"] = "log2(count+1) — inherited from upstream"
        h5f.attrs["orientation"]   = "samples_as_rows"
        h5f.attrs["n_samples"]     = n_samples
        h5f.attrs["n_genes"]       = n_genes
    del arr_T

    write_min = (time.time() - t) / 60
    size_gb = dst_gex.stat().st_size / 1e9
    log(f"      Wrote {size_gb:.2f} GB to {dst_gex}  ({write_min:.1f} min)")

    return dst_gex


def main():
    log("=" * 70)
    log("CSV → HDF5 conversion (Scope B, memory-safe)")
    log("=" * 70)
    log(f"  Source:      {SRC_DIR}")
    log(f"  Destination: {DST_DIR}")

    # Sanity
    for p in [SRC_DIR / "train/gex.csv", SRC_DIR / "train/clin.csv",
              SRC_DIR / "test/gex.csv",  SRC_DIR / "test/clin.csv"]:
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")

    t_total = time.time()
    train_h5 = convert_split("train")
    test_h5  = convert_split("test")
    total_min = (time.time() - t_total) / 60

    log("\n" + "=" * 70)
    log(f"DONE — HDF5 conversion complete  ({total_min:.1f} min total)")
    log("=" * 70)
    log(f"  Train HDF5: {train_h5}  ({train_h5.stat().st_size / 1e9:.2f} GB)")
    log(f"  Test  HDF5: {test_h5}   ({test_h5.stat().st_size / 1e9:.2f} GB)")
    log("\nNext steps:")
    log("  1. Verify HDF5 files (h5dump or python h5py read)")
    log("  2. Run h5_dataset.py + train_denoising_vae_h5.py "
        "(I'll write these next)")
    log("=" * 70)


if __name__ == "__main__":
    main()
