"""
h5_dataloader.py - HDF5-backed DataImporter for Flexynesis.

The stock Flexynesis DataImporter reads each modality with pd.read_csv, which
loads values as float64 by default. For large feature matrices this doubles
the in-memory footprint and makes data loading the dominant memory cost of a
training run.

H5DataImporter is a drop-in subclass of DataImporter that loads any modality
from HDF5 as native float32 when an .h5 file is present, and otherwise falls
back to the standard CSV path. All other Flexynesis behaviour - data cleanup,
NaN imputation, label encoding, scaling, and torch dataset construction - is
inherited unchanged.

HDF5 files are expected in the layout produced by csv_to_h5.py:
    /matrix         (n_samples, n_features) float32
    /sample_ids     (n_samples,) byte strings
    /feature_names  (n_features,) byte strings

Each modality folder is expected to contain clin.csv (as required by the
stock DataImporter); every other modality may be supplied as either an .h5
file or a .csv file.

Usage:
    from flexynesis.h5_dataloader import H5DataImporter
    di = H5DataImporter(path='my_dataset', data_types=['gex'])
    train_ds, test_ds = di.import_data()
"""
import os

import h5py
import numpy as np
import pandas as pd

from flexynesis.data import DataImporter


class H5DataImporter(DataImporter):
    """
    DataImporter subclass that loads modality matrices from HDF5 when
    available, and from CSV otherwise.

    Only read_data() and validate_data_folders() are overridden; the parent
    DataImporter and all downstream Flexynesis components are unchanged.

    HDF5 files are expected in the layout written by csv_to_h5.py:
        /matrix         (n_samples, n_features) float32
        /sample_ids     (n_samples,) byte strings
        /feature_names  (n_features,) byte strings
    """

    def read_data(self, folder_path):
        """
        Override of DataImporter.read_data.

        Returns a dict {file_name: DataFrame} in the format Flexynesis
        expects. Each modality in self.data_types is loaded from an .h5 file
        if present, otherwise from the corresponding .csv. clin.csv is always
        loaded from CSV.
        """
        print("\n[INFO] ----------------- Reading Data (HDF5) ----------------- ")
        data = {}
        required_files = {'clin.csv'} | {f"{dt}.csv" for dt in self.data_types}

        for file in required_files:
            file_name = os.path.splitext(file)[0]

            # Modality matrices may be HDF5; clin.csv is always CSV.
            if file_name in self.data_types:
                h5_path = os.path.join(folder_path, f"{file_name}.h5")
                if not os.path.exists(h5_path):
                    # Fall back to CSV when no HDF5 file is present.
                    csv_path = os.path.join(folder_path, file)
                    print(f"[INFO] HDF5 not found at {h5_path}, "
                          f"falling back to CSV: {csv_path}")
                    data[file_name] = pd.read_csv(csv_path, index_col=0)
                else:
                    print(f"[INFO] Importing {h5_path} (HDF5)...")
                    data[file_name] = self._read_h5_as_dataframe(h5_path)
            else:
                csv_path = os.path.join(folder_path, file)
                print(f"[INFO] Importing {csv_path}...")
                data[file_name] = pd.read_csv(csv_path, index_col=0)

        return data

    @staticmethod
    def _read_h5_as_dataframe(h5_path):
        """
        Read an HDF5 modality file into a DataFrame matching the Flexynesis
        CSV convention:
            index   = feature identifiers (str)
            columns = sample identifiers  (str)
            values  = float32

        HDF5 stores the matrix samples-as-rows; it is transposed here to
        features-as-rows during DataFrame construction.
        """
        with h5py.File(h5_path, 'r') as f:
            n_samples, n_features = f['matrix'].shape
            print(f"[INFO]   HDF5 shape: {n_samples:,} samples x "
                  f"{n_features:,} features (float32)")

            arr = f['matrix'][:]  # (n_samples, n_features) float32
            sample_ids = [s.decode() for s in f['sample_ids'][:]]
            feature_names = [g.decode() for g in f['feature_names'][:]]

        # Transpose to (n_features, n_samples) to match the CSV convention.
        # np.ascontiguousarray forces a real copy in C-order.
        arr_t = np.ascontiguousarray(arr.T)
        del arr

        df = pd.DataFrame(arr_t, index=feature_names, columns=sample_ids)
        print(f"[INFO]   DataFrame ready: {df.shape}  "
              f"dtype: {df.dtypes.iloc[0]}")
        return df

    def validate_data_folders(self, training_path, testing_path):
        """
        Override to accept either a .csv or an .h5 file for each modality in
        self.data_types. clin.csv must still be present as CSV.
        """
        for split_name, path in [("training", training_path),
                                 ("testing", testing_path)]:
            if not os.path.isdir(path):
                raise ValueError(
                    f"{split_name} folder does not exist: {path}")
            missing = []
            if not os.path.exists(os.path.join(path, "clin.csv")):
                missing.append("clin.csv")
            for dt in self.data_types:
                h5_ok = os.path.exists(os.path.join(path, f"{dt}.h5"))
                csv_ok = os.path.exists(os.path.join(path, f"{dt}.csv"))
                if not (h5_ok or csv_ok):
                    missing.append(f"{dt}.h5 or {dt}.csv")
            if missing:
                raise ValueError(
                    f"Missing files in {split_name} folder: "
                    f"{', '.join(missing)}")
        print("[INFO] Validating data folders... OK (HDF5 or CSV accepted)")
