"""
h5_dataloader.py — HDF5-backed DataImporter for Flexynesis.

Why this exists:
  Flexynesis stock DataImporter calls pd.read_csv(gex.csv) which loads
  as float64 (default), doubling memory: 14 GB CSV → 28 GB DataFrame +
  Flexynesis copies → 60+ GB peak → ROG GUI freeze.

  This subclass overrides ONLY read_data() to load gex from HDF5 as
  native float32, halving memory and skipping the slow CSV parse.

  Everything else (cleanup, scaling, label encoding, torch dataset
  construction) uses Flexynesis's tested infrastructure unchanged.

Usage:
  from h5_dataloader import H5DataImporter
  di = H5DataImporter(
      path='processed_scaled_411k_tissue_B_h5',
      data_types=['gex'],
      log_transform=False,
      top_percentile=100,
      min_features=100,
  )
  train_ds, test_ds = di.import_data()
  # train_ds, test_ds are standard Flexynesis MultiomicsDataset objects.

Expected directory layout:
  {path}/train/gex.h5      ← HDF5 (samples × genes, float32)
  {path}/train/clin.csv    ← regular CSV (small, no issue)
  {path}/test/gex.h5
  {path}/test/clin.csv

Memory profile:
  HDF5 → float32 numpy array : ~8 GB for 118k × 16k train
  DataFrame wrap (no copy)   : ~8 GB stable
  Flexynesis normalize/scale : ~12 GB transient
  PEAK: ~20 GB (vs 60+ with stock CSV path) → safe on 64 GB ROG
"""
import os
import h5py
import numpy as np
import pandas as pd

from flexynesis.data import DataImporter


class H5DataImporter(DataImporter):
    """
    Subclasses Flexynesis DataImporter; overrides read_data to use HDF5
    for the gex modality (the big one), CSV for everything else.

    Expects gex stored as HDF5 with this layout:
        /expression    (n_samples, n_genes) float32
        /sample_ids    (n_samples,) bytes — sample IDs in HDF5 row order
        /gene_symbols  (n_genes,)   bytes — gene symbols in HDF5 col order
    """

    def read_data(self, folder_path):
        """
        Override of DataImporter.read_data.

        Returns dict {file_name: DataFrame} in the same format Flexynesis
        expects, but loads `gex` from HDF5 (float32) instead of CSV (float64).
        Other modalities and clin.csv still load from CSV.
        """
        print("\n[INFO] ----------------- Reading Data (HDF5) ----------------- ")
        data = {}
        required_files = {'clin.csv'} | {f"{dt}.csv" for dt in self.data_types}

        for file in required_files:
            file_name = os.path.splitext(file)[0]

            # GEX → HDF5 path; everything else → CSV
            if file_name in self.data_types:
                h5_path = os.path.join(folder_path, f"{file_name}.h5")
                if not os.path.exists(h5_path):
                    # Fall back to CSV if HDF5 missing (graceful degradation)
                    csv_path = os.path.join(folder_path, file)
                    print(f"[INFO] HDF5 not found at {h5_path}, "
                          f"falling back to CSV: {csv_path}")
                    data[file_name] = pd.read_csv(csv_path, index_col=0)
                else:
                    print(f"[INFO] Importing {h5_path} (HDF5)...")
                    data[file_name] = self._read_h5_as_dataframe(h5_path)
            else:
                # clin.csv etc — load as usual
                csv_path = os.path.join(folder_path, file)
                print(f"[INFO] Importing {csv_path}...")
                data[file_name] = pd.read_csv(csv_path, index_col=0)

        return data

    @staticmethod
    def _read_h5_as_dataframe(h5_path):
        """
        Read HDF5 gex into a DataFrame matching Flexynesis CSV convention:
            index   = gene symbols (str)
            columns = sample IDs   (str)
            values  = float32

        HDF5 stores samples-as-rows (118k × 16k); we transpose to
        genes-as-rows during DataFrame construction.

        Memory: peak ~16 GB during transpose (118k × 16k × 4 = 7.7 GB,
        plus one transpose copy). Drops to ~8 GB after.
        """
        with h5py.File(h5_path, 'r') as f:
            n_samples, n_genes = f['expression'].shape
            print(f"[INFO]   HDF5 shape: {n_samples:,} samples × {n_genes:,} genes (float32)")

            # Read as samples-rows, then transpose
            arr = f['expression'][:]  # (n_samples, n_genes) float32
            sample_ids   = [s.decode() for s in f['sample_ids'][:]]
            gene_symbols = [g.decode() for g in f['gene_symbols'][:]]

        # Transpose to (n_genes, n_samples) to match Flexynesis CSV convention
        # Note: np.ascontiguousarray forces real copy in C-order
        arr_T = np.ascontiguousarray(arr.T)
        del arr

        df = pd.DataFrame(
            arr_T,
            index=gene_symbols,
            columns=sample_ids,
        )
        print(f"[INFO]   DataFrame ready: {df.shape}  dtype: {df.dtypes.iloc[0]}")
        return df

    def validate_data_folders(self, training_path, testing_path):
        """
        Override to accept either .csv OR .h5 for data_types modalities.
        clin.csv must still exist as CSV.
        """
        import os
        for split_name, path in [("training", training_path), ("testing", testing_path)]:
            if not os.path.isdir(path):
                raise ValueError(f"{split_name} folder does not exist: {path}")
            missing = []
            if not os.path.exists(os.path.join(path, "clin.csv")):
                missing.append("clin.csv")
            for dt in self.data_types:
                h5_ok = os.path.exists(os.path.join(path, f"{dt}.h5"))
                csv_ok = os.path.exists(os.path.join(path, f"{dt}.csv"))
                if not (h5_ok or csv_ok):
                    missing.append(f"{dt}.h5 or {dt}.csv")
            if missing:
                raise ValueError(f"Missing files in {split_name} folder: {', '.join(missing)}")
        print("[INFO] Validating data folders... OK (HDF5 or CSV accepted)")
