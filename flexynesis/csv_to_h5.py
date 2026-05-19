#!/usr/bin/env python3
"""
csv_to_h5.py - Convert a feature matrix CSV to HDF5 for memory-efficient loading.

Large feature-matrix CSV files are slow to parse and, when read with pandas,
are stored as float64 by default - doubling their in-memory footprint.
Converting such a CSV to HDF5 once allows it to be loaded later as native
float32 with substantially lower peak memory.

The input CSV is expected to have features as rows and samples as columns, with
the first column containing feature identifiers and the header row containing
sample identifiers. This matches the layout used elsewhere in Flexynesis.

The output HDF5 file has the following layout:
    /matrix         (n_samples, n_features) float32, chunked (1, n_features)
    /sample_ids     (n_samples,) byte strings
    /feature_names  (n_features,) byte strings
    attrs: created_by, source_csv, orientation, n_samples, n_features

Data are stored samples-as-rows with per-row chunking so that individual
samples can be read efficiently. H5DataImporter reads files in this layout.

Usage:
    python -m flexynesis.csv_to_h5 input.csv output.h5
    python -m flexynesis.csv_to_h5 input.csv output.h5 --chunksize 1000
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEFAULT_CHUNKSIZE = 500  # rows (features) read per pandas chunk


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def convert_csv_to_h5(src_csv, dst_h5, chunksize=DEFAULT_CHUNKSIZE):
    """
    Convert a single feature-matrix CSV to HDF5.

    Parameters
    ----------
    src_csv : str or Path
        Input CSV: features as rows, samples as columns, first column is the
        feature index.
    dst_h5 : str or Path
        Output HDF5 path. Parent directories are created if needed.
    chunksize : int
        Number of CSV rows parsed per chunk while streaming the file.

    Returns
    -------
    Path
        The path to the written HDF5 file.
    """
    src_csv = Path(src_csv)
    dst_h5 = Path(dst_h5)

    if not src_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {src_csv}")

    dst_h5.parent.mkdir(parents=True, exist_ok=True)

    log(f"Converting {src_csv} -> {dst_h5}")

    # ---------- Scan structure: sample IDs (header) and feature IDs (col 0) ----------
    header_df = pd.read_csv(src_csv, nrows=0, index_col=0)
    sample_ids = header_df.columns.tolist()
    n_samples = len(sample_ids)

    feature_col = pd.read_csv(src_csv, usecols=[0])
    feature_names = feature_col.iloc[:, 0].astype(str).tolist()
    n_features = len(feature_names)
    del feature_col

    log(f"  {n_samples:,} samples x {n_features:,} features")

    # ---------- Stream CSV into a pre-allocated (n_features, n_samples) array ----------
    arr = np.empty((n_features, n_samples), dtype=np.float32)

    chunks = pd.read_csv(src_csv, index_col=0, chunksize=chunksize)
    row_idx = 0
    for chunk in chunks:
        chunk_arr = chunk.values.astype(np.float32, copy=False)
        n = chunk_arr.shape[0]
        arr[row_idx:row_idx + n] = chunk_arr
        row_idx += n
        del chunk_arr

    if row_idx != n_features:
        raise ValueError(
            f"Row count mismatch: scanned {n_features} features, "
            f"read {row_idx} while streaming."
        )

    # ---------- Transpose to (n_samples, n_features) for samples-as-rows storage ----------
    arr_t = np.ascontiguousarray(arr.T)
    del arr

    # ---------- Write HDF5 ----------
    with h5py.File(dst_h5, "w") as h5f:
        # Per-row chunking: one chunk == one sample, for fast single-sample reads.
        h5f.create_dataset("matrix", data=arr_t, chunks=(1, n_features))
        h5f.create_dataset("sample_ids", data=np.array(sample_ids, dtype="S"))
        h5f.create_dataset("feature_names",
                           data=np.array(feature_names, dtype="S"))
        h5f.attrs["created_by"] = "csv_to_h5.py"
        h5f.attrs["source_csv"] = str(src_csv)
        h5f.attrs["orientation"] = "samples_as_rows"
        h5f.attrs["n_samples"] = n_samples
        h5f.attrs["n_features"] = n_features
    del arr_t

    size_gb = dst_h5.stat().st_size / 1e9
    log(f"  Wrote {size_gb:.2f} GB to {dst_h5}")
    return dst_h5


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a feature-matrix CSV to HDF5 for "
                    "memory-efficient loading."
    )
    parser.add_argument("input_csv", help="Input CSV (features x samples).")
    parser.add_argument("output_h5", help="Output HDF5 file path.")
    parser.add_argument(
        "--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
        help=f"CSV rows parsed per chunk (default: {DEFAULT_CHUNKSIZE}).",
    )
    args = parser.parse_args(argv)

    try:
        convert_csv_to_h5(args.input_csv, args.output_h5, args.chunksize)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
