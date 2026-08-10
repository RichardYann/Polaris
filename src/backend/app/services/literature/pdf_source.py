"""Generic and safe PDF source handling for user supplied / metadata URLs."""

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx

from app.core.config import get_settings

MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_REDIRECTS = 5


class PdfValidationError(ValueError):
    """The downloaded/uploaded payload is not an acceptable PDF."""


class PdfUrlFetchError(RuntimeError):
    """A remote PDF URL could not be fetched safely."""


def validate_pdf_bytes(content: bytes) -> None:
    if not content.startswith(b"%PDF-"):
        raise PdfValidationError("文件不是有效的 PDF（缺少 %PDF- 文件头）")
    if len(content) > MAX_PDF_BYTES:
        raise PdfValidationError("PDF 超过 100 MB 大小限制")


async def _validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise PdfUrlFetchError("仅支持公开的 http/https PDF 链接")
    if parsed.username or parsed.password:
        raise PdfUrlFetchError("PDF 链接不能包含用户名或密码")
    try:
        literal = ipaddress.ip_address(parsed.hostname)
        addresses = [literal]
    except ValueError:
        try:
            rows = await asyncio.get_running_loop().getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError as e:
            raise PdfUrlFetchError(f"无法解析 PDF 地址：{parsed.hostname}") from e
        addresses = list({ipaddress.ip_address(row[4][0]) for row in rows})
    if not addresses or any(not address.is_global for address in addresses):
        raise PdfUrlFetchError("PDF 链接不能指向本机、内网或保留地址")


async def download_pdf_url(url: str) -> bytes:
    """Download a public PDF with redirect, SSRF and size protections."""
    current = url.strip()
    settings = get_settings()
    async with httpx.AsyncClient(
        proxy=settings.outbound_proxy or None,
        timeout=httpx.Timeout(60.0, connect=15.0),
        follow_redirects=False,
        headers={"User-Agent": "Polaris/1.0 PDF fetcher"},
    ) as client:
        for redirect_count in range(MAX_REDIRECTS + 1):
            await _validate_public_url(current)
            try:
                async with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise PdfUrlFetchError("PDF 源返回了无目标的重定向")
                        if redirect_count >= MAX_REDIRECTS:
                            raise PdfUrlFetchError("PDF 链接重定向次数过多")
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    parts: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_PDF_BYTES:
                            raise PdfValidationError("PDF 超过 100 MB 大小限制")
                        parts.append(chunk)
            except (PdfValidationError, PdfUrlFetchError):
                raise
            except httpx.HTTPError as e:
                raise PdfUrlFetchError(f"PDF 下载失败：{type(e).__name__}: {e}") from e
            content = b"".join(parts)
            validate_pdf_bytes(content)
            return content
    raise PdfUrlFetchError("PDF 下载失败")


async def download_paper_pdf(paper: object) -> bytes:
    """Resolve a paper PDF: arXiv first, then the Semantic Scholar OA URL."""
    arxiv_id = getattr(paper, "arxiv_id", None)
    if arxiv_id:
        from app.services.literature import get_arxiv_client

        content = await get_arxiv_client().download_pdf(str(arxiv_id))
        validate_pdf_bytes(content)
        return content
    external_ids = getattr(paper, "external_ids", None) or {}
    pdf_url = external_ids.get("pdf_url") if isinstance(external_ids, dict) else None
    if pdf_url:
        return await download_pdf_url(str(pdf_url))
    raise PdfUrlFetchError("论文没有可自动获取的 PDF 来源")
