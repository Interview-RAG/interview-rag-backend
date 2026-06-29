from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "Interview Preparation Q&A Collection", 0, 1, "C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

import textwrap

def generate_pdf(qa_list, output_path="collection.pdf"):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for index, qa in enumerate(qa_list, start=1):
        questions = qa.get("questions", [])
        primary_question = questions[0] if questions else "Question?"
        
        # Format and encode question
        pdf.set_font("Arial", "B", 12)
        question_text = f"{index}. {primary_question}"
        question_text = question_text.encode('latin-1', 'replace').decode('latin-1')
        
        # Safely wrap and print question lines
        for orig_line in question_text.split('\n'):
            wrapped_q = textwrap.wrap(orig_line, width=90)
            if not wrapped_q:
                pdf.ln(7)
            for line in wrapped_q:
                pdf.cell(0, 7, txt=line, ln=1)
        
        # Format and encode answer
        pdf.set_font("Arial", "", 12)
        answer_text = f"==> {qa.get('answer', '')}"
        answer_text = answer_text.encode('latin-1', 'replace').decode('latin-1')
        
        # Safely wrap and print answer lines with indentation
        for orig_line in answer_text.split('\n'):
            wrapped_a = textwrap.wrap(orig_line, width=85)
            if not wrapped_a:
                pdf.ln(7)
            for line in wrapped_a:
                pdf.set_x(20) # Indent the answer
                pdf.cell(0, 7, txt=line, ln=1)
        
        pdf.ln(8) # Space between QA pairs

    pdf.output(output_path)
    return output_path
