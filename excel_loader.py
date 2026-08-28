# src/excel_loader.py

import os
import pandas as pd
from src.column_detection import ColumnDetector

class DataLoader:
    def __init__(self):
        self.detector = ColumnDetector()

    def load_and_merge(self, file_paths):
        """
        Accepts a list of file paths (CSV or Excel), standardizes their 
        columns, tracks file origins, and returns a single merged pandas DataFrame.
        """
        dataframes = []
        all_metadata_cols = set()
        all_recognized_cols = set()

        for path in file_paths:
            if not os.path.exists(path):
                continue
                
            # Read format based on extension
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            
            # Clean trailing spaces from data or column headers
            df.columns = df.columns.astype(str).str.strip()
            for c in df.columns:
                if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                    df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
            
            # Run column recognition framework
            df, recognized, metadata = self.detector.standardize_headers(df)
            
            # Record trackable metadata origin
            df['Source File'] = os.path.basename(path)
            
            dataframes.append(df)
            all_metadata_cols.update(metadata)
            all_recognized_cols.update(recognized)

        if not dataframes:
            return pd.DataFrame(), [], []

        # Merge all dataframes seamlessly
        merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        return merged_df, list(all_recognized_cols), list(all_metadata_cols)
