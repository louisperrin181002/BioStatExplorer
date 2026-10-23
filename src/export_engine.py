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

    @staticmethod
    def save_figure_to(figure, path, format_type="png", dpi=600):
        """Saves the figure at *path*. SVG/PDF keep text as text (editable
        in Inkscape/Illustrator); PNG/TIFF at *dpi*."""
        import matplotlib
        rc = {"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}
        kw = {}
        if format_type in ("tif", "tiff"):
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        with matplotlib.rc_context(rc):
            figure.savefig(path, format=format_type, dpi=dpi,
                           bbox_inches="tight", **kw)
        return path

    @staticmethod
    def save_excel_summary_to(path, summary_df, stat_report_text,
                              comparisons_df=None):
        """Excel workbook: group summary, every comparison as a table, and
        the full text report."""
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary Statistics",
                                index=False)
            if comparisons_df is not None and not comparisons_df.empty:
                comparisons_df.to_excel(writer, sheet_name="Comparisons",
                                        index=False)
            pd.DataFrame({"Statistical Analysis Report":
                          stat_report_text.split("\n")}).to_excel(
                writer, sheet_name="Report", index=False)
        return path
