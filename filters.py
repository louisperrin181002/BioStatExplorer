# src/filters.py

class FilterEngine:
    @staticmethod
    def get_unique_values(df, column):
        """Returns a sorted list of unique categorical options in a column."""
        if df is None or column not in df.columns:
            return []
        return sorted(df[column].dropna().unique().astype(str))

    @staticmethod
    def apply_categorical_filter(df, column, selected_values):
        """
        Filters the dataframe to only include rows matching selected values.
        If no values are selected, it acts as an 'All' pass-through.
        """
        if df is None or df.empty or not column or not selected_values:
            return df
        
        # Cast column to string to safely match UI selection strings
        return df[df[column].astype(str).isin(selected_values)]

    @staticmethod
    def apply_multi_filter(df, filter_map):
        """
        Applies several simultaneous per-column filters (AND across columns,
        OR within a column's selected values) — e.g. filter_map =
        {"Sample": ["WT", "LP112"], "Organism": ["P. fluo"]}.
        A column with an empty/None value list is treated as 'no restriction'.
        """
        if df is None or df.empty or not filter_map:
            return df
        result = df
        for column, selected_values in filter_map.items():
            if not selected_values or column not in result.columns:
                continue
            result = result[result[column].astype(str).isin(selected_values)]
        return result
