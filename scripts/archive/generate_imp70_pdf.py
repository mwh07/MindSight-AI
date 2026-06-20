# generate_imp68_pdf.py
import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def construct_questionnaire_pdf():
    csv_source = "imp68_questionnaire.csv"
    output_pdf = "IMP68_Blank_Questionnaire.pdf"
    
    if not os.path.exists(csv_source):
        from generate_imp68_csv import build_imp68_questionnaire_csv
        build_imp68_questionnaire_csv()

    print("[IMP-70] Compiling High-Fidelity Blank Questionnaire PDF at Root...")
    df = pd.read_csv(csv_source)
    
    # 0.5-inch compact printing layout boundary mapping
    doc = SimpleDocTemplate(
        output_pdf, pagesize=letter, 
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=colors.HexColor("#0F172A"), spaceAfter=3)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=11, textColor=colors.HexColor("#475569"), spaceAfter=12)
    domain_heading_style = ParagraphStyle('DomainHead', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=colors.HexColor("#1E3A8A"), spaceBefore=12, spaceAfter=5, keepWithNext=True)
    cell_id_style = ParagraphStyle('CellID', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor("#334155"))
    cell_text_style = ParagraphStyle('CellText', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor("#0F172A"))
    cell_meta_style = ParagraphStyle('CellMeta', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=7.5, leading=9, textColor=colors.HexColor("#64748B"))

    story = []
    story.append(Paragraph("MINDSIGHT INTEGRATED CLINICAL BATTERY (IMP-70 PLATINUM)", title_style))
    story.append(Paragraph("<b>Instructions:</b> Complete all data fields. Response metrics are normalized, dynamically inverted via vector direction attributes, and transformed cyclically before entering core tensor stacks.", subtitle_style))
    
    current_domain = ""
    table_data = []
    
    # Structural column width configurations (Total printable area width = 540)
    col_widths = [65, 335, 140]
    
    for _, row in df.iterrows():
        dom = row['Domain']
        if dom != current_domain:
            if table_data:
                t = Table(table_data, colWidths=col_widths, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F8FAFC")),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                    ('TOPPADDING', (0,0), (-1,-1), 4),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
                ]))
                story.append(t)
                table_data = []
            current_domain = dom
            story.append(Paragraph(str(current_domain).upper(), domain_heading_style))
            table_data.append([
                Paragraph("<b>ID</b>", cell_id_style), 
                Paragraph("<b>Diagnostic Inquiry Column</b>", cell_id_style), 
                Paragraph("<b>Input Target / Bounds</b>", cell_id_style)
            ])
            
        p_id = Paragraph(str(row['Question_ID']), cell_id_style)
        
        # Build composite cell text array including explicit vector direction tags
        meta_string = f"[ {row['Response_Type']} | Vector: {row['Scoring_Direction']} ]"
        text_cell_block = [
            Paragraph(str(row['Question_Text']), cell_text_style), 
            Spacer(1, 1), 
            Paragraph(meta_string, cell_meta_style)
        ]
        table_data.append([p_id, text_cell_block, ""])
        
    if table_data:
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F8FAFC")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        story.append(t)

    doc.build(story)
    print("🎉 Success! Beautifully formatted layout built at root as 'IMP68_Blank_Questionnaire.pdf'")

if __name__ == "__main__":
    construct_questionnaire_pdf()