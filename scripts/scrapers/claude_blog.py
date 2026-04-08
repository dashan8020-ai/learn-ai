"""
claude_blog.py — 爬取 Claude Blog 页面。

目标页面: https://claude.com/blog
提取所有 /blog/<slug> 格式的文章链接。
"""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup

NAME = "Claude Blog"
SLUG = "claude-blog"
CATEGORY = "industry"

_URL = "https://claude.com/blog"
_LINK_RE = re.compile(r"^/blog/[\w-]+$")
_HEADERS = {"User-Agent": "Mozilla/5.0 learn-ai-bot"}


def scrape() -> list[dict]:
    """爬取 Claude Blog，返回条目列表。"""
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
        if not prev or len(text) < len(prev):
            best[href] = text

    entries: list[dict] = []
    for href, title in best.items():
        link = f"https://claude.com{href}"
        entries.append({
            "title": title,
            "link": link,
            "summary": "",
            "source": NAME,
            "slug": SLUG,
            "cat": CATEGORY,
        })
    return entries
