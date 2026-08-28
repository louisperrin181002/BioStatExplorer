# src/export_engine.py
import os
import pandas as pd

class ExportEngine:
    @staticmethod
    def save_figure(figure, directory, filename, format_type="png", dpi=600):
        """Saves active plot at publication-quality high resolution (600 DPI)."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        full_path = os.path.join(directory, f"{filename}.{format_type}")
        # bbox_inches='tight' prevents axis labels from getting cut off
        figure.savefig(full_path, format=format_type, dpi=dpi, bbox_inches='tight')
        return full_path

    @staticmethod
    def save_excel_summary(summary_df, stat_report_text, directory, filename):
        """Exports clean Excel files containing processed stats summaries."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        full_path = os.path.join(directory, f"{filename}.xlsx")
        
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            # Sheet 1: Summary Table Data
            summary_df.to_excel(writer, sheet_name="Summary Statistics", index=False)
            
            # Sheet 2: Raw text structural report from Prism Inference engine
            lines = stat_report_text.split('\n')
            report_df = pd.DataFrame({"Statistical Analysis Output": lines})
            report_df.to_excel(writer, sheet_name="Hypothesis Tests", index=False)
            
        return full_path
