"""
pdf_utils.py
------------
Small helper for pulling text out of an uploaded resume PDF.
Kept in its own file so app.py stays easy to read.
"""

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFExtractionError(Exception):
    """Raised when we can't get usable text out of the uploaded PDF."""
    pass


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Takes a Streamlit UploadedFile object and returns the extracted text.
    Raises PDFExtractionError with a beginner-friendly message on failure.
    """
    try:
        reader = PdfReader(uploaded_file)
    except PdfReadError:
        raise PDFExtractionError(
            "This file doesn't look like a valid, readable PDF. "
            "Please try re-saving/exporting it and upload again."
        )
    except Exception:
        raise PDFExtractionError(
            "We couldn't open this PDF. It may be corrupted or password-protected."
        )

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            raise PDFExtractionError(
                "This PDF is password-protected. Please upload an unlocked copy."
            )

    text_parts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        text_parts.append(page_text)

    full_text = "\n".join(text_parts).strip()

    if not full_text:
        raise PDFExtractionError(
            "We couldn't find any selectable text in this PDF. It might be a "
            "scanned image rather than a text-based document. Try pasting the "
            "resume text directly instead."
        )

    return full_text
