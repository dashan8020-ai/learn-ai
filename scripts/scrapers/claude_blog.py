"""
claude_blog.py — 爬取 Claude Blog 页面。

目标页面: https://claude.com/blog
提取所有 /blog/<slug> 格式的文章链接，支持按日期增量过滤。
"""

from __future__ import annotations

import datetime as dt
import re

import httpx
from bs4 import BeautifulSoup, Tag

NAME = "Claude Blog"
SLUG = "claude-blog"
CATEGORY = "industry"

_URL = "https://claude.com/blog"
_LINK_RE = re.compile(r"^/blog/[\w-]+$")
_CTA_RE = re.compile(
    r"^(Read more|Learn more|See more|View more|View|Try Claude|Contact sales)$",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"((?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct"
    r"|Nov|Dec)\s+\d{1,2},?\s+\d{4})"
)
_DATE_FMTS = ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y")
_HEADERS = {"User-Agent": "Mozilla/5.0 learn-ai-bot"}


def _parse_date(text: str) -> dt.datetime | None:
    """尝试多种格式解析日期文本。"""
    for fmt in _DATE_FMTS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _find_date_near(a_tag: Tag) -> dt.datetime | None:
    """从链接的父级元素中查找日期文本。"""
    node = a_tag.parent
    for _ in range(8):
        if node is None:
            break
        text = node.get_text(" ", strip=True)
        m = _DATE_RE.search(text)
        if m:
            return _parse_date(m.group(1))
        node = node.parent
    return None


def scrape(since: dt.datetime | None = None) -> list[dict]:
    """爬取 Claude Blog，返回条目列表。

    Args:
        since: 只返回该时间之后的文章。None 表示不过滤。
    """
    resp = httpx.get(_URL, follow_redirects=True, timeout=30, headers=_HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 第一遍：收集每个 href 的最佳标题 + 日期
    best: dict[str, str] = {}
    dates: dict[str, dt.datetime | None] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not _LINK_RE.match(href):
            continue
        text = a.get_text(" ", strip=True)
        if len(text) < 4 or _CTA_RE.match(text):
            continue
        prev = best.get(href, "")
        if not prev or (len(text) > len(prev) and len(text) <= 200):
            best[href] = text
        # 只在第一次遇到时解析日期
        if href not in dates:
            dates[href] = _find_date_near(a)

    entries: list[dict] = []
    for href, title in best.items():
        # 增量过滤：如果解析到了日期且早于 since，跳过
        if since and dates.get(href) and dates[href] < since:
            continue
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
