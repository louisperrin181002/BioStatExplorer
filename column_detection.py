# src/column_detection.py

class ColumnDetector:
    def __init__(self):
        # Maps user-defined column name variations to standardized internal names
        self.translation_map = {
            # Green fluorescent proteins
            'gfp': 'GFP', 'sfgfp': 'GFP', 'green': 'GFP',
            # Red fluorescent proteins
            'mkate': 'mKate', 'rfp': 'mKate', 'red': 'mKate',
            # Wild types
            'wt': 'WT', 'wildtype': 'WT', 'wild_type': 'WT',
            # Optical density
            'od': 'Optical Density', 'od600': 'Optical Density', 'abs': 'Optical Density',
            # Media conditions
            'media': 'Medium', 'medium': 'Medium', 'broth': 'Medium',
            # Target biology
            'organism': 'Organism', 'species': 'Organism', 'strain': 'Organism'
        }

    def standardize_headers(self, df):
        """
        Renames known columns using semantic dictionary.
        Matches whole words/tokens inside a header (so "Value GFP" or
        "GFP_raw" both resolve to "GFP"), not just an exact full-string match.
        Returns the modified dataframe, a list of recognized targets, and metadata targets.
        """
        import re

        rename_dict = {}
        recognized = []
        metadata = []
        used_standard_names = set()

        for col in df.columns:
            cleaned_col = str(col).strip().lower()
            tokens = re.split(r'[\s_\-/]+', cleaned_col)
            tokens = [t for t in tokens if t]

            standard_name = None
            # 1. Try an exact full-header match first (preserves old behavior)
            if cleaned_col in self.translation_map:
                standard_name = self.translation_map[cleaned_col]
            else:
                # 2. Fall back to matching any individual token, e.g.
                #    "Value GFP" -> tokens ["value", "gfp"] -> "gfp" matches
                for token in tokens:
                    if token in self.translation_map:
                        standard_name = self.translation_map[token]
                        break

            if standard_name is not None:
                # Avoid collapsing two distinct raw columns onto the same
                # standardized name (e.g. a file with both "GFP" and "gfp")
                proposed = standard_name
                suffix = 2
                while proposed in used_standard_names and proposed != col:
                    proposed = f"{standard_name} ({suffix})"
                    suffix += 1
                used_standard_names.add(proposed)
                rename_dict[col] = proposed
                if proposed not in recognized:
                    recognized.append(proposed)
            else:
                # Unknown columns naturally cascade into metadata filters
                rename_dict[col] = col
                metadata.append(col)

        standardized_df = df.rename(columns=rename_dict)
        return standardized_df, recognized, metadata
