"""
generate_friendly_guide.py
==========================
Generates a super clear, friendly, story-like PDF guide titled
"AgriVision_Friendly_Story_Guide.pdf" that explains the entire project using
simple real-world analogies, zero scary formulas, and clean plain English
so ANY student or friend can instantly understand how it works and how it was built.
"""

import os
from fpdf import FPDF
from datetime import datetime

def sanitize_pdf_text(text: str) -> str:
    """Replaces Unicode characters with Latin-1 safe equivalents for FPDF."""
    if not isinstance(text, str):
        return str(text)
    replacements = {
        "—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
        "…": "...", "°": " deg ", "•": "*", "→": "->", "≥": ">=", "≤": "<=",
        "✅": "[OK]", "⚠️": "[Warning]", "🧪": "[Chem]", "🌿": "[Organic]",
        "🛡️": "[Prevent]", "🔍": "[Inspect]", "💡": "[Tip]", "🧠": "[AI]",
        "📊": "[Stats]", "💬": "[Chat]", "📄": "[PDF]", "🌾": "[Crop]"
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin-1", "ignore").decode("latin-1")

class FriendlyGuidePDF(FPDF):
    def header(self):
        self.set_fill_color(16, 44, 34)  # Deep Emerald
        self.rect(0, 0, 210, 22, 'F')
        
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 4)
        self.cell(190, 7, sanitize_pdf_text("AGRIVISION AI: THE SIMPLE & FRIENDLY EXPLANATION GUIDE"), align="C")
        
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(167, 243, 208)
        self.set_xy(10, 12)
        self.cell(190, 5, sanitize_pdf_text("How We Built AgriVision AI Explained in Plain English (Zero Complex Math!)"), align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, sanitize_pdf_text(f"AgriVision AI Friendly Story Guide | Page {self.page_no()} of {{nb}} | Group 113 Major Project"), align="C")

def build_friendly_guide_pdf(output_filename="AgriVision_Friendly_Story_Guide.pdf"):
    pdf = FriendlyGuidePDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(12, 26, 12)
    pdf.set_auto_page_break(auto=True, margin=16)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 1: THE REAL-WORLD STORY & WHY REGULAR AI FAILS
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(16, 44, 34)
    pdf.cell(0, 8, sanitize_pdf_text("Part 1: The Real Story - Why Did We Build This?"), ln=True)
    pdf.set_draw_color(52, 211, 153)
    pdf.line(12, 34, 198, 34)
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(30, 41, 59)
    p1 = (
        "Imagine a farmer standing in a field in a small village. He notices dark spots spreading on his tomato leaves.\n"
        "He is worried: 'Is this a fungus? Is it a virus? Should I buy spray A or spray B?'\n\n"
        "If he guesses wrong and buys the wrong chemical:\n"
        "  - He loses his money.\n"
        "  - The chemical burns his crops.\n"
        "  - His entire harvest for the year is destroyed.\n\n"
        "Real expert crop doctors (agronomists) are very far away. So farmers want to take a photo on their mobile phone "
        "and get an instant diagnosis. But regular AI apps FAIL terribly in real farming! Here are the 3 reasons why:"
    )
    pdf.multi_cell(186, 4.8, sanitize_pdf_text(p1))
    pdf.ln(4)

    reasons = [
        ("1. The 'Blind Doctor' Problem (No Visual Proof)",
         "Regular AI gives a label like 'Tomato Disease' but DOES NOT show where the disease is on the leaf! "
         "The farmer has no idea if the AI looked at the disease spot or if it just looked at background grass."),

        ("2. The 'Cocky AI' Problem (Fake 99% Confidence)",
         "Regular AI is super overconfident. If you accidentally take a picture of your shoe, your hand, or a tractor, "
         "a regular AI will NOT say 'I don't know'. It will yell 'Tomato Late Blight - 99% Confident!' because it is forced to guess!"),

        ("3. The 'Lying AI' Problem (Fake Chemical Dosages)",
         "If you ask regular ChatGPT for chemical advice, it often hallucinates. It might invent a fake chemical name or "
         "tell the farmer to spray 10x too much chemical, which poisons the soil!")
    ]

    for title, desc in reasons:
        pdf.set_fill_color(254, 242, 242)
        pdf.set_draw_color(252, 165, 165)
        pdf.rect(12, pdf.get_y(), 186, 22, 'DF')
        pdf.set_xy(15, pdf.get_y() + 2)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(153, 27, 27)
        pdf.cell(0, 4.5, sanitize_pdf_text(title), ln=True)
        pdf.set_font("Helvetica", "", 8.8)
        pdf.set_text_color(127, 29, 29)
        pdf.set_x(15)
        pdf.multi_cell(180, 4.0, sanitize_pdf_text(desc))
        pdf.ln(4)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 2: HOW WE FIXED ALL 3 PROBLEMS (OUR 4 SMART TOOLS)
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(16, 44, 34)
    pdf.cell(0, 8, sanitize_pdf_text("Part 2: How AgriVision AI Fixes These 3 Problems"), ln=True)
    pdf.set_draw_color(52, 211, 153)
    pdf.line(12, 34, 198, 34)
    pdf.ln(4)

    solutions = [
        ("Smart Tool 1: ResNet50 & EfficientNet AI Brains",
         "We trained two smart AI models on 54,000 leaf photos across 38 crop types.\n"
         "  * ResNet50: The accuracy champion (94.87% accurate).\n"
         "  * EfficientNetV2: The mobile champion (92.20% accurate, 2x faster, uses 77% less phone memory)."),

        ("Smart Tool 2: Temperature Scaling (The 'Calm Down' Filter)",
         "We added a simple mathematical filter (T = 1.1959) that calms the AI down.\n"
         "Instead of screaming '99% confident' on tricky photos, it calculates honest confidence scores.\n"
         "Result: Reduced fake confidence errors by 61.6%!"),

        ("Smart Tool 3: The Safety Rejection Guardrail",
         "We set a strict rule: If confidence is below 60%, OR if the photo is a hand/shoe/machinery,\n"
         "the app STOPS and says: 'Warning: I am uncertain. Please consult a real human agronomist before spraying.'"),

        ("Smart Tool 4: Grad-CAM Heatmap (The Highlighter Pen)",
         "When the AI makes a diagnosis, it draws a glowing red/yellow heatmap over the exact disease spots on the leaf photo.\n"
         "The farmer can SEE with their own eyes exactly what the AI examined!"),

        ("Smart Tool 5: Knowledge-Grounded RAG (The Reference Textbook)",
         "Instead of letting AI guess chemical remedies, we built an official 'Agri Dictionary' (rag_knowledge_base.json).\n"
         "When the farmer asks for care steps, the AI reads the official dictionary FIRST, then answers.\n"
         "Result: 100% verified advice with ZERO fake chemical dosages!")
    ]

    for title, desc in solutions:
        pdf.set_fill_color(240, 253, 244)
        pdf.set_draw_color(187, 247, 208)
        pdf.rect(12, pdf.get_y(), 186, 26, 'DF')
        pdf.set_xy(15, pdf.get_y() + 2)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(20, 83, 45)
        pdf.cell(0, 4.5, sanitize_pdf_text(title), ln=True)
        pdf.set_font("Helvetica", "", 8.8)
        pdf.set_text_color(30, 41, 59)
        pdf.set_x(15)
        pdf.multi_cell(180, 4.0, sanitize_pdf_text(desc))
        pdf.ln(4)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 3: HOW WE BUILT THIS STEP-BY-STEP (DEVELOPER JOURNEY)
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(16, 44, 34)
    pdf.cell(0, 8, sanitize_pdf_text("Part 3: How We Built This Step-by-Step (Developer Journey)"), ln=True)
    pdf.set_draw_color(52, 211, 153)
    pdf.line(12, 34, 198, 34)
    pdf.ln(4)

    dev_steps = [
        ("Step 1: Getting & Splitting the Leaf Data",
         "We gathered 54,305 leaf photos. We split them carefully into 3 buckets:\n"
         "  * Bucket 1 (Study Set - 70%): 37,998 photos used to teach the AI.\n"
         "  * Bucket 2 (Practice Set - 15%): 8,161 photos used to adjust settings.\n"
         "  * Bucket 3 (Final Exam Set - 15%): 8,146 photos kept hidden for final testing."),

        ("Step 2: Training the AI Brains",
         "We took pre-trained AI brains that already recognized shapes (ImageNet) and trained them on leaf patterns.\n"
         "  * ResNet50 scored 94.87% on the final exam.\n"
         "  * EfficientNetV2 scored 92.20% and ran twice as fast."),

        ("Step 3: Creating the Agronomy Dictionary",
         "We created rag_knowledge_base.json covering all 38 diseases. For every disease, we wrote down:\n"
         "  - Exact Symptoms\n  - Verified Chemical Spray & Dosage\n  - Organic / Neem Oil Remedies\n  - Prevention Steps"),

        ("Step 4: Building the Interactive Web App",
         "We built a clean website using Streamlit (App.py):\n"
         "  - Added model selection toggles (ResNet50 vs EfficientNetV2).\n"
         "  - Added Grad-CAM heatmap visualization.\n"
         "  - Added voice commands in Hindi & English.\n"
         "  - Added a button to download a printable PDF report card.")
    ]

    for title, desc in dev_steps:
        pdf.set_font("Helvetica", "B", 9.8)
        pdf.set_text_color(20, 83, 45)
        pdf.cell(0, 5, sanitize_pdf_text(title), ln=True)
        pdf.set_font("Helvetica", "", 8.8)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(186, 4.2, sanitize_pdf_text(desc))
        pdf.ln(3)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 4: HOW TO DEMO THIS TO YOUR FRIEND & PROFESSORS
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(16, 44, 34)
    pdf.cell(0, 8, sanitize_pdf_text("Part 4: How to Explain & Demo This to Anyone in 2 Minutes"), ln=True)
    pdf.set_draw_color(52, 211, 153)
    pdf.line(12, 34, 198, 34)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("Your 2-Minute Explanation Script:"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    script_text = (
        "\"Hey! Let me show you AgriVision AI. It is an intelligent crop doctor app built for farmers.\n\n"
        "Here is what makes it special:\n"
        "1. It classifies 38 plant diseases with up to 94.87% accuracy using ResNet50 and EfficientNetV2.\n"
        "2. It uses Temperature Scaling so it never gives fake 99% confidence on non-leaf photos.\n"
        "3. It draws a glowing Grad-CAM heatmap over the leaf so you can SEE exact disease spots.\n"
        "4. It uses RAG to look up an official agronomist reference book before answering, guaranteeing 100% accurate chemical dosages with ZERO AI hallucinations!\"\n"
    )
    pdf.multi_cell(186, 4.6, sanitize_pdf_text(script_text))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("Live Demo Steps:"), ln=True)
    pdf.set_font("Helvetica", "", 8.8)
    demo = [
        ("1. Open Web App", "Type `streamlit run App.py` in terminal."),
        ("2. Upload Leaf Photo", "Drag and drop any plant leaf image."),
        ("3. View Heatmap & Diagnosis", "See instant diagnosis, calibrated confidence %, and Grad-CAM heatmap."),
        ("4. Ask Voice Questions", "Speak in Hindi or English to get care advice."),
        ("5. Download PDF Card", "Click 'Download PDF Diagnostic Report' for a printable report.")
    ]
    for title, desc in demo:
        pdf.set_font("Helvetica", "B", 9.0)
        pdf.set_text_color(20, 83, 45)
        pdf.cell(40, 4.5, sanitize_pdf_text(title + ": "), 0)
        pdf.set_font("Helvetica", "", 8.8)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(146, 4.5, sanitize_pdf_text(desc), ln=True)

    # Signoff Box
    pdf.ln(8)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(12, pdf.get_y(), 186, 20, 'DF')
    pdf.set_xy(16, pdf.get_y() + 3)
    pdf.set_font("Helvetica", "B", 8.8)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(85, 4.5, sanitize_pdf_text("AgriVision AI Project - Group 113"), 0)
    pdf.cell(85, 4.5, sanitize_pdf_text("Supervisor: Ms. Pooja Deswal"), ln=True)
    pdf.set_x(16)
    pdf.set_font("Helvetica", "", 8.2)
    pdf.cell(85, 4.5, sanitize_pdf_text("Team: Harsh Gupta, Manthan, Sumit Kumar"), 0)
    pdf.cell(85, 4.5, sanitize_pdf_text("JSS Academy of Technical Education, Noida"), ln=True)

    output_path = os.path.join(os.getcwd(), output_filename)
    pdf.output(output_path)
    print(f"Friendly Story Guide PDF generated successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    build_friendly_guide_pdf()
