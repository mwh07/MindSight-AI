import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Complete schema matrix containing exactly 70 features matching schema definitions
questions_data = [
    # Demographic Baseline
    {"id": "age", "domain": "DEMOGRAPHIC BASELINE", "text": "What is your current chronological age in years?", "type": "numeric", "label": "[Continuous Numeric Integer Entry]"},
    {"id": "gender", "domain": "DEMOGRAPHIC BASELINE", "text": "What is your biological sex assigned at birth?", "type": "choice", "label": "[Discrete Check Box: (0) Male  (1) Female]"},
    
    # Domain 1: Personality
    {"id": "EXT1", "domain": "DOMAIN 1: PERSONALITY", "text": "I am the life of the party.", "type": "likert5"},
    {"id": "EXT2", "domain": "DOMAIN 1: PERSONALITY", "text": "I don't talk a lot.", "type": "likert5"},
    {"id": "EXT3", "domain": "DOMAIN 1: PERSONALITY", "text": "I feel comfortable around people.", "type": "likert5"},
    {"id": "EST1", "domain": "DOMAIN 1: PERSONALITY", "text": "I get stressed out easily.", "type": "likert5"},
    {"id": "EST2", "domain": "DOMAIN 1: PERSONALITY", "text": "I am relaxed most of the time.", "type": "likert5"},
    {"id": "EST3", "domain": "DOMAIN 1: PERSONALITY", "text": "I worry about things.", "type": "likert5"},
    {"id": "AGR1", "domain": "DOMAIN 1: PERSONALITY", "text": "I feel little concern for others.", "type": "likert5"},
    {"id": "AGR2", "domain": "DOMAIN 1: PERSONALITY", "text": "I am interested in people.", "type": "likert5"},
    {"id": "AGR3", "domain": "DOMAIN 1: PERSONALITY", "text": "I feel sympathy for others' feelings.", "type": "likert5"},
    {"id": "CSN1", "domain": "DOMAIN 1: PERSONALITY", "text": "I am always prepared.", "type": "likert5"},
    {"id": "CSN2", "domain": "DOMAIN 1: PERSONALITY", "text": "I leave my duties undone.", "type": "likert5"},
    {"id": "CSN3", "domain": "DOMAIN 1: PERSONALITY", "text": "I pay attention to details.", "type": "likert5"},
    {"id": "OPN1", "domain": "DOMAIN 1: PERSONALITY", "text": "I have a rich vocabulary.", "type": "likert5"},
    {"id": "OPN2", "domain": "DOMAIN 1: PERSONALITY", "text": "I have difficulty understanding abstract ideas.", "type": "likert5"},
    {"id": "OPN3", "domain": "DOMAIN 1: PERSONALITY", "text": "I have a vivid imagination.", "type": "likert5"},
    
    # Domain 2: Self-Esteem
    {"id": "Q1", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "I feel that I am a person of worth, at least on an equal plane with others.", "type": "likert5"},
    {"id": "Q2", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "I feel that I have a number of good qualities.", "type": "likert5"},
    {"id": "Q3", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "All in all, I am inclined to feel that I am a failure.", "type": "likert5"},
    {"id": "Q4", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "I am able to do things as well as most other people.", "type": "likert5"},
    {"id": "Q5", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "I feel I do not have much to be proud of.", "type": "likert5"},
    {"id": "Q6", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "I take a positive attitude toward myself.", "type": "likert5"},
    {"id": "Q7", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "On the whole, I am satisfied with myself.", "type": "likert5"},
    {"id": "Q8", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "I wish I could have more respect for myself.", "type": "likert5"},
    {"id": "Q9", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "I certainly feel useless at times.", "type": "likert5"},
    {"id": "Q10", "domain": "DOMAIN 2: SELF-ESTEEM", "text": "At times I think I am no good at all.", "type": "likert5"},
    
    # Domain 3: Mood and Sleep
    {"id": "DPQ010", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "Little interest or pleasure in doing things over the past 2 weeks.", "type": "likert4_mood"},
    {"id": "DPQ020", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "Feeling down, depressed, or hopeless over the past 2 weeks.", "type": "likert4_mood"},
    {"id": "DPQ030", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "Trouble falling or staying asleep, or sleeping too much.", "type": "likert4_mood"},
    {"id": "DPQ040", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "Feeling tired or having little energy.", "type": "likert4_mood"},
    {"id": "DPQ050", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "Poor appetite or overeating patterns.", "type": "likert4_mood"},
    {"id": "DPQ060", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "Feeling bad about yourself, or that you are a failure.", "type": "likert4_mood"},
    {"id": "DPQ070", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "Trouble concentrating on basic tasks (e.g., reading or watching TV).", "type": "likert4_mood"},
    {"id": "DPQ080", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "Moving/speaking so slowly or quickly that others noticed.", "type": "likert4_mood"},
    {"id": "DPQ090", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "Thoughts that you would be better off dead, or of hurting yourself.", "type": "likert4_mood"},
    {"id": "DPQ100", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "How difficult have these mood problems made it to work or get along with people?", "type": "likert4_sev"},
    {"id": "SLQ300", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "What time do you usually go to bed on workdays/weekdays?", "type": "time", "label": "[Format String (HH:MM)]"},
    {"id": "SLQ310", "domain": "DOMAIN 3: MOOD AND SLEEP", "text": "What time do you usually wake up on workdays/weekdays?", "type": "time", "label": "[Format String (HH:MM)]"},
    
    # Domain 4: Digital and Social Risk
    {"id": "IAT1", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you stay online longer than you originally intended?", "type": "likert5_internet"},
    {"id": "IAT2", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you neglect household/daily tasks to spend more time online?", "type": "likert5_internet"},
    {"id": "IAT3", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you prefer the excitement of the internet over real-world relationships?", "type": "likert5_internet"},
    {"id": "IAT4", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you form new digital relationships with online users?", "type": "likert5_internet"},
    {"id": "IAT5", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do close connections complain to you about your internet usage?", "type": "likert5_internet"},
    {"id": "IAT6", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do your grades, productivity, or work responsibilities suffer from screen time?", "type": "likert5_internet"},
    {"id": "IAT7", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you check electronic communication channels before executing required tasks?", "type": "likert5_internet"},
    {"id": "IAT8", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often does your job performance diminish due to distracting online activity?", "type": "likert5_internet"},
    {"id": "IAT9", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you become defensive or highly secretive regarding your browser behaviors?", "type": "likert5_internet"},
    {"id": "IAT10", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you mask disturbing real-life thoughts using soothing internet media?", "type": "likert5_internet"},
    {"id": "loneliness1", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you feel completely 'in tune' with the people around you?", "type": "likert4_lone"},
    {"id": "loneliness2", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you feel that you severely lack deep companionship?", "type": "likert4_lone"},
    {"id": "loneliness3", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you feel that there is truly no one you can turn to?", "type": "likert4_lone"},
    {"id": "loneliness4", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you feel isolated and entirely alone?", "type": "likert4_lone"},
    {"id": "loneliness5", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you feel strongly integrated as part of a group of friends?", "type": "likert4_lone"},
    {"id": "loneliness6", "domain": "DOMAIN 4: DIGITAL AND SOCIAL RISK", "text": "How often do you feel that you have a significant amount in common with those around you?", "type": "likert4_lone"},
    
    # Domain 5: Occupational Burnout
    {"id": "work_hours_per_week", "domain": "DOMAIN 5: OCCUPATIONAL BURNOUT", "text": "Average total number of active working hours per week.", "type": "numeric", "label": "[Numeric Entry: ______ Hours]"},
    {"id": "meetings_per_day", "domain": "DOMAIN 5: OCCUPATIONAL BURNOUT", "text": "Average total number of corporate meetings attended daily.", "type": "numeric", "label": "[Numeric Entry: ______ Meetings]"},
    {"id": "work_life_balance_score", "domain": "DOMAIN 5: OCCUPATIONAL BURNOUT", "text": "Rate your subjective overall work-life balance satisfaction level.", "type": "score10", "label": "[1-10 Discrete Scale (1=Poor, 10=Excellent)]"},
    {"id": "job_satisfaction_score", "domain": "DOMAIN 5: OCCUPATIONAL BURNOUT", "text": "Rate your subjective professional and career fulfillment level.", "type": "score10", "label": "[1-10 Discrete Scale (1=Poor, 10=Excellent)]"},
    {"id": "deadline_pressure_score", "domain": "DOMAIN 5: OCCUPATIONAL BURNOUT", "text": "Rate the frequency/severity of time constraints and urgency pressures.", "type": "score10", "label": "[1-10 Intensity Scale (1=Low, 10=Extreme)]"},
    {"id": "autonomy_score", "domain": "DOMAIN 5: OCCUPATIONAL BURNOUT", "text": "Rate your level of control and decision freedom over execution tasks.", "type": "score10", "label": "[1-10 Intensity Scale (1=None, 10=Total)]"},
    {"id": "stress_score", "domain": "DOMAIN 5: OCCUPATIONAL BURNOUT", "text": "Rate the baseline cumulative stress experienced over the past month.", "type": "score10", "label": "[1-10 Intensity Scale (1=Low, 10=Extreme)]"},
    {"id": "social_support_score", "domain": "DOMAIN 5: OCCUPATIONAL BURNOUT", "text": "Rate the perceived strength of your immediate workplace support framework.", "type": "score10", "label": "[1-10 Strength Scale (1=Weak, 10=Strong)]"},
    
    # Domain 6: Severe Clinical
    {"id": "unwanted_thoughts", "domain": "DOMAIN 6: SEVERE CLINICAL", "text": "Experiencing recurrent, distressing, intrusive thoughts or images.", "type": "binary"},
    {"id": "repetitive_behaviors", "domain": "DOMAIN 6: SEVERE CLINICAL", "text": "Feeling compelled to repeat physical actions or rigid mental rituals.", "type": "binary"},
    {"id": "overthinking", "domain": "DOMAIN 6: SEVERE CLINICAL", "text": "Excessive rumination over insignificant daily micro-interactions.", "type": "binary"},
    {"id": "mind_going_blank", "domain": "DOMAIN 6: SEVERE CLINICAL", "text": "Cognitive paralysis or loss of memory continuity during stress situations.", "type": "binary"},
    {"id": "avoidance_social_activity", "domain": "DOMAIN 6: SEVERE CLINICAL", "text": "Active avoidance of social events, crowds, or public areas out of distress.", "type": "binary"},
    {"id": "panic", "domain": "DOMAIN 6: SEVERE CLINICAL", "text": "Sudden, unprovoked surges of overwhelming physical terror or heart palpitations.", "type": "binary"},
    {"id": "hypervigilance", "domain": "DOMAIN 6: SEVERE CLINICAL", "text": "Continuous high-alert monitoring of surroundings to guard against threats.", "type": "binary"}
]

class NumberedCanvas(canvas.Canvas):
    """ Canvas class supporting a precise professional page number tracker (Page X of Y) """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Upper Running Header (Skips page 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "MINDSIGHT INTEGRATED CLINICAL BATTERY (IMP-70)")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 742, letter[0]-54, 742)
            
        # Standardized Lower running footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, footer_text)
        self.drawString(54, 36, "CONFIDENTIAL — CLINICAL & ACADEMIC DATA CONTRACT TARGET")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0]-54, 48)
        self.restoreState()

def generate_pdf(output_filename="IMP70_Questionnaire.pdf"):
    # Target simple letter geometry with clear 0.75 in margins
    doc = SimpleDocTemplate(output_filename, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#1A365D")   # Deep Slate Blue
    text_color = colors.HexColor("#2D3748")      # Charcoal Body Text
    light_bg = colors.HexColor("#F7FAFC")        # Soft row variance background
    
    # Typographic definitions
    title_style = ParagraphStyle('MainTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=primary_color, alignment=1, spaceAfter=4)
    subtitle_style = ParagraphStyle('SubTitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor("#4A5568"), alignment=1, spaceAfter=14)
    instr_style = ParagraphStyle('Instr', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9, leading=13, textColor=colors.HexColor("#4A5568"), spaceAfter=12)
    domain_style = ParagraphStyle('DomHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=colors.white, keepWithNext=True)
    q_text_style = ParagraphStyle('QText', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12, textColor=text_color)
    q_id_style = ParagraphStyle('QID', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=12, textColor=primary_color)
    bubble_style = ParagraphStyle('Bubble', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, alignment=1)

    story = []
    
    # Document Header Generation
    story.append(Paragraph("MINDSIGHT INTEGRATED CLINICAL BATTERY (IMP-70)", title_style))
    story.append(Paragraph("Unified Psychological Registration Contract (Structural Schema Variant 2.3)", subtitle_style))
    story.append(Paragraph("<b>Instructions:</b> Provide responses for all items. Shaded tracking elements feed parameter variances directly into downstream modeling architectures.", instr_style))
    
    current_domain = None
    domain_elements = []
    
    for idx, q in enumerate(questions_data, start=1):
        if q['domain'] != current_domain:
            if domain_elements:
                story.append(KeepTogether(domain_elements))
                domain_elements = []
            
            current_domain = q['domain']
            
            # Colored Structural Block Header
            hdr_table = Table([[Paragraph(f"&nbsp;&nbsp;{current_domain}", domain_style)]], colWidths=[letter[0]-108])
            hdr_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), primary_color),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            domain_elements.append(Spacer(1, 10))
            domain_elements.append(hdr_table)
            domain_elements.append(Spacer(1, 5))
            
            # Context Legends depending on active domains
            if "PERSONALITY" in current_domain or "SELF-ESTEEM" in current_domain:
                domain_elements.append(Paragraph("<b>Scale:</b> [ 1 = Disagree ]   [ 2 = Slightly Disagree ]   [ 3 = Neutral ]   [ 4 = Slightly Agree ]   [ 5 = Agree ]", instr_style))
            elif "MOOD AND SLEEP" in current_domain:
                domain_elements.append(Paragraph("<b>Scale:</b> [ 0 = Not at all ]   [ 1 = Several days ]   [ 2 = More than half the days ]   [ 3 = Nearly every day ]", instr_style))
            elif "DIGITAL AND SOCIAL" in current_domain:
                domain_elements.append(Paragraph("<b>Scale Metrics:</b> IAT items range [1 = Rarely] to [5 = Always]. Loneliness metrics range [1 = Never] to [4 = Often].", instr_style))

        id_cell = Paragraph(q['id'], q_id_style)
        txt_cell = Paragraph(f"{idx}. {q['text']}", q_text_style)
        
        # Layout metrics based on feature specifications
        if q['type'] in ["likert5", "likert5_internet"]:
            row = [id_cell, txt_cell, Paragraph("[ 1 ]", bubble_style), Paragraph("[ 2 ]", bubble_style), Paragraph("[ 3 ]", bubble_style), Paragraph("[ 4 ]", bubble_style), Paragraph("[ 5 ]", bubble_style)]
            widths = [45, 239, 44, 44, 44, 44, 44]
        elif q['type'] in ["likert4_lone", "likert4_mood", "likert4_sev"]:
            row = [id_cell, txt_cell, Paragraph("[ 0 / 1 ]", bubble_style), Paragraph("[ 1 / 2 ]", bubble_style), Paragraph("[ 2 / 3 ]", bubble_style), Paragraph("[ 3 / 4 ]", bubble_style), ""]
            widths = [45, 239, 55, 55, 55, 55, 0]
        elif q['type'] == "binary":
            row = [id_cell, txt_cell, Paragraph("[ No &nbsp;0 ]", bubble_style), Paragraph("[ Yes &nbsp;1 ]", bubble_style), "", "", ""]
            widths = [45, 239, 110, 110, 0, 0, 0]
        else:
            row = [id_cell, txt_cell, Paragraph(f"<b>{q.get('label', '[ Entry Box ]')}</b>", bubble_style), "", "", "", ""]
            widths = [45, 239, 220, 0, 0, 0, 0]
            
        q_table = Table([row], colWidths=widths)
        bg = light_bg if idx % 2 == 0 else colors.white
        
        q_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LINEBELOW', (0,0), (-1,-1), 0.3, colors.HexColor("#E2E8F0"))
        ]))
        domain_elements.append(q_table)
        
    if domain_elements:
        story.append(KeepTogether(domain_elements))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"File successfully exported to: {output_filename}")

if __name__ == "__main__":
    generate_pdf()