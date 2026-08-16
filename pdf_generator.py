import os
from fpdf import FPDF
from datetime import datetime

def sanitize_pdf_text(text: str) -> str:
    """Replaces Unicode characters (em-dash, smart quotes, emojis) with Latin-1 safe characters."""
    if not isinstance(text, str):
        return str(text)
    replacements = {
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "°": " deg ",
        "✅": "[Healthy]",
        "⚠️": "[Warning]",
        "🧪": "[Chemical]",
        "🌿": "[Organic]",
        "🛡️": "[Prevention]",
        "🔍": "[Diagnosis]",
        "📋": "[Protocol]"
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Encode to latin-1 dropping unrepresentable characters
    return text.encode("latin-1", "ignore").decode("latin-1")

class PlantHealthPDFReport(FPDF):
    def header(self):
        self.set_fill_color(11, 29, 22)  # Dark agritech green
        self.rect(0, 0, 210, 25, 'F')
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "AgriVision AI - Plant Health Diagnostic Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

def create_pdf_report(
    disease_name: str,
    confidence_score: float,
    severity_info: dict,
    rag_context: dict,
    care_instructions: str = "",
    language: str = "English"
) -> str:
    """Generates an official downloadable PDF health report."""
    disease_name = sanitize_pdf_text(disease_name)
    severity_level = sanitize_pdf_text(severity_info['severity_level'])
    urgency = sanitize_pdf_text(severity_info['urgency'])
    
    symptoms = sanitize_pdf_text(rag_context.get("symptoms", "N/A"))
    chem_treatment = sanitize_pdf_text(rag_context.get("chemical_treatment", "N/A"))
    org_remedy = sanitize_pdf_text(rag_context.get("organic_remedy", "N/A"))
    prevention = sanitize_pdf_text(rag_context.get("prevention", "N/A"))

    pdf = PlantHealthPDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Section
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, "1. Executive Diagnostic Summary", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Key Metrics Table
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(50, 8, "Crop Condition:", border=0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, disease_name, ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(50, 8, "Model Confidence:", border=0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"{confidence_score * 100:.2f}%", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(50, 8, "Infection Severity:", border=0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"{severity_level} ({severity_info['affected_percentage']}% leaf area)", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(50, 8, "Urgency Level:", border=0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, urgency, ln=True)

    # Section 2: RAG Knowledge Base Treatments
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, "2. RAG Certified Agronomist Care Protocol", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Typical Symptoms Observed:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, symptoms)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Chemical Treatment & Dosage:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, chem_treatment)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Organic & Bio-Remedies:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, org_remedy)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Preventative Actions:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, prevention)

    output_path = os.path.join(os.path.dirname(__file__), "Plant_Health_Report.pdf")
    pdf.output(output_path)
    return output_path
