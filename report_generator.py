# src/report_generator.py

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class PDFReportGenerator:
    @staticmethod
    def generate_pdf(output_path, summary_df, stat_text, temp_plot_path):
        """
        Assembles a publication-ready multi-page report containing project overview,
        the generated graph, summary statistics, and hypothesis outcomes.
        """
        # Page Setup
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                                rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []
        styles = getSampleStyleSheet()

        # Custom Lab Styles
        title_style = ParagraphStyle(
            'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=24,
            textColor=colors.HexColor('#2C3E50'), spaceAfter=15
        )
        section_style = ParagraphStyle(
            'SectionHeader', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14,
            textColor=colors.HexColor('#2980B9'), spaceBefore=15, spaceAfter=8
        )
        body_style = ParagraphStyle(
            'ReportBody', parent=styles['BodyText'], fontName='Helvetica', fontSize=10,
            leading=14, textColor=colors.HexColor('#333333')
        )

        # Title Block
        story.append(Paragraph("BioStat Explorer — Analytical Report", title_style))
        story.append(Spacer(1, 10))

        # 1. Figure Insertion
        story.append(Paragraph("1. Generated Data Visualization", section_style))
        if os.path.exists(temp_plot_path):
            # Read plot file and scale nicely for a letter page width
            img = Image(temp_plot_path, width=360, height=270)
            story.append(img)
        else:
            story.append(Paragraph("[Visual Graph Not Detected]", body_style))
        story.append(Spacer(1, 15))

        # 2. Descriptive Summary Table
        story.append(Paragraph("2. Group Summary Statistics", section_style))
        if summary_df is not None and not summary_df.empty:
            # Build structurally aligned array list for layout processing
            table_data = [list(summary_df.columns)] + summary_df.values.tolist()
            
            # Format numbers safely for scannability
            for row in table_data[1:]:
                for idx, val in enumerate(row):
                    if isinstance(val, (int, float)):
                        row[idx] = f"{val:.4f}" if isinstance(val, float) else str(val)

            t = Table(table_data, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#34495E')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 9),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8F9FA')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 9),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t)
        else:
            story.append(Paragraph("No descriptive stats computed.", body_style))
        story.append(Spacer(1, 15))

        # 3. Structural Hypothesis Verification Data
        story.append(Paragraph("3. Hypothesis Testing Matrix & Insights", section_style))
        formatted_text = stat_text.replace('\n', '<br/>')
        story.append(Paragraph(formatted_text, body_style))

        # Build Document
        doc.build(story)
