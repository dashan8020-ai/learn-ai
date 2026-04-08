"""
anthropic.py — 爬取 Anthropic News 页面。

目标页面: https://www.anthropic.com/news
提取所有 /news/<slug> 格式的文章链接。
"""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup

NAME = "Anthropic"
SLUG = "anthropic"
CATEGORY = "industry"

_URL = "https://www.anthropic.com/news"
_LINK_RE = re.compile(r"^/news/[\w-]+$")
_HEADERS = {"User-Agent": "Mozilla/5.0 learn-ai-bot"}


def scrape() -> list[dict]:
    """爬取 Anthropic News，返回条目列表。"""
    resp = httpx.get(_URL, follow_redirects=True, timeout=30, headers=_HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 收集每个匹配链接对应的最佳标题
    best: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not _LINK_RE.match(href):
            continue
        text = a.get_text(" ", strip=True)
        if len(text) < 4:
            continue
        prev = best.get(href, "")
        # 选最短且非空的文本作为标题（避免抓到大段描述）
        if not prev or len(text) < len(prev):
            best[href] = text

    entries: list[dict] = []
    for href, title in best.items():
        link = f"https://www.anthropic.com{href}"
        entries.append({
            "title": title,
            "link": link,
            "summary": "",
            "source": NAME,
            "slug": SLUG,
            "cat": CATEGORY,
        })
    return entries
