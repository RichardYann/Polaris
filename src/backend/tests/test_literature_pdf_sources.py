"""Regressions for Corpus ID import and safe PDF source selection."""

from types import SimpleNamespace

import pytest

from app.services import paper_import
from app.services.literature import pdf_source


class _S2Stub:
    async def get_paper(self, paper_id: str):
        assert paper_id == "CorpusId:13756489"
        return {
            "paperId": "s2-paper-id",
            "title": "A Corpus Paper",
            "authors": [{"name": "Alice"}],
            "year": 2025,
            "externalIds": {"ArXiv": "2501.01234", "DOI": "10.1/example"},
            "openAccessPdf": {"url": "https://publisher.example/paper.pdf"},
        }


async def test_corpus_id_maps_s2_metadata_and_arxiv(monkeypatch):
    monkeypatch.setattr(paper_import, "get_s2_client", lambda: _S2Stub())
    fields = await paper_import._fields_from_corpus_id("CorpusId:13756489")
    assert fields["source"] == "semantic_scholar"
    assert fields["arxiv_id"] == "2501.01234"
    assert fields["external_ids"] == {
        "s2": "s2-paper-id",
        "corpus_id": "13756489",
        "arxiv": "2501.01234",
        "doi": "10.1/example",
        "pdf_url": "https://publisher.example/paper.pdf",
    }


async def test_pdf_resolution_prefers_arxiv_over_s2_url(monkeypatch):
    calls: list[str] = []

    class _ArxivStub:
        async def download_pdf(self, arxiv_id: str) -> bytes:
            calls.append(arxiv_id)
            return b"%PDF-arxiv"

    async def fail_if_s2_used(_url: str) -> bytes:
        raise AssertionError("S2 PDF URL must not be used when an arXiv id exists")

    import app.services.literature as literature

    monkeypatch.setattr(literature, "get_arxiv_client", lambda: _ArxivStub())
    monkeypatch.setattr(pdf_source, "download_pdf_url", fail_if_s2_used)
    paper = SimpleNamespace(
        arxiv_id="2501.01234", external_ids={"pdf_url": "https://s2.example/paper.pdf"}
    )
    assert await pdf_source.download_paper_pdf(paper) == b"%PDF-arxiv"
    assert calls == ["2501.01234"]


async def test_pdf_url_rejects_private_address():
    with pytest.raises(pdf_source.PdfUrlFetchError):
        await pdf_source.download_pdf_url("http://127.0.0.1/private.pdf")
