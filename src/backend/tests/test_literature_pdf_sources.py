"""Regressions for uploaded and remotely fetched PDF sources."""

import pytest

from app.services.literature import pdf_source


def test_pdf_validation_accepts_pdf_header():
    pdf_source.validate_pdf_bytes(b"%PDF-1.7\ncontent")


def test_pdf_validation_rejects_non_pdf_payload():
    with pytest.raises(pdf_source.PdfValidationError):
        pdf_source.validate_pdf_bytes(b"<html>not a pdf</html>")


async def test_pdf_url_rejects_private_address():
    with pytest.raises(pdf_source.PdfUrlFetchError):
        await pdf_source.download_pdf_url("http://127.0.0.1/private.pdf")
