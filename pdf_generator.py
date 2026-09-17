from fpdf import FPDF
import textwrap


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "PrepAI Study Guide", 0, 1, "C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def _latin1(text):
    """fpdf's core fonts are latin-1 only; drop what will not encode."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _write_wrapped(pdf, text, width, indent=None):
    for orig_line in _latin1(text).split("\n"):
        wrapped = textwrap.wrap(orig_line, width=width)
        if not wrapped:
            pdf.ln(7)
        for line in wrapped:
            if indent is not None:
                pdf.set_x(indent)
            pdf.cell(0, 7, txt=line, ln=1)


def generate_pdf(sections, output_path="collection.pdf"):
    """`sections` is [{"tag": str, "items": [{"questions": [...], "answer": str}]}].

    A plain list of Q&A pairs is also accepted, and prints as one untitled run.
    """
    if sections and "tag" not in sections[0]:
        sections = [{"tag": None, "items": sections}]

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for section in sections:
        tag = section.get("tag")
        if tag:
            pdf.set_font("Arial", "B", 14)
            pdf.ln(2)
            pdf.cell(0, 9, txt=_latin1(tag.upper()), ln=1)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(pdf.get_x(), pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)

        for index, qa in enumerate(section.get("items") or [], start=1):
            questions = qa.get("questions") or []
            primary_question = questions[0] if questions else "Question?"

            pdf.set_font("Arial", "B", 12)
            _write_wrapped(pdf, f"{index}. {primary_question}", width=90)

            pdf.set_font("Arial", "", 12)
            _write_wrapped(pdf, f"==> {qa.get('answer', '')}", width=85, indent=20)

            pdf.ln(8)

    pdf.output(output_path)
    return output_path
