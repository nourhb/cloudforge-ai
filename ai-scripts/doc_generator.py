from transformers import pipeline
from typing import Tuple
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

_summarizer = None

def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline('summarization', model='sshleifer/tiny-t5', framework='pt')
    return _summarizer

def _write_pdf(text: str, out_path: str) -> None:
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    x = 40
    y = height - 40
    for line in text.splitlines():
        if y < 60:
            c.showPage()
            y = height - 40
        c.drawString(x, y, line[:110])
        y -= 14
    c.save()

def generate_docs(code: str) -> Tuple[str, str]:
    """Generate Markdown and PDF from code/comments.
    Returns (markdown_text, pdf_path)
    """
    if not code:
        md = "# Documentation\n\nNo code provided."
        pdf_path = os.path.abspath('docs/generated_docs.pdf')
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        _write_pdf(md, pdf_path)
        return md, pdf_path

    # Summarize code using tiny model for speed
    text = code[:4000]
    summary = _get_summarizer()(text, max_length=128, min_length=32, do_sample=False)[0]['summary_text']

    md = f"""
# Auto-Generated Documentation

## Summary
{summary}

## Original Code (excerpt)
```\n{text}\n```
""".strip()

    pdf_path = os.path.abspath('docs/generated_docs.pdf')
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    # Convert markdown to plaintext for PDF
    plaintext = markdown.markdown(md)
    # strip HTML tags quickly
    import re
    plain = re.sub('<[^<]+?>', '', plaintext)
    _write_pdf(plain, pdf_path)
    return md, pdf_path

# TEST: generate_docs returns md and creates PDF
