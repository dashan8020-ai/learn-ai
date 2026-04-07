# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "feedparser>=6.0",
#     "httpx>=0.27",
#     "openai>=1.0",
# ]
# ///
"""
daily_update.py — 每日自动拉取 AI 社区/行业动态，生成 journal 条目并更新 model-tracker。

用法:
    uv run scripts/daily_update.py                # 使用 LLM 摘要（需设 OPENAI_API_KEY）
    uv run scripts/daily_update.py --no-llm       # 仅拉取 RSS，不调用 LLM

工作流程:
    1. 从预配置的 RSS 源拉取过去 24h 内的新条目
    2. (可选) 调用 LLM 将条目分类并生成中文摘要
    3. 写入 journal/YYYY/MM/DD.md
    4. 若有模型发布相关条目，追加到 landscape/model-tracker.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import textwrap
from pathlib import Path

import feedparser
import httpx

# ──────────────────────────────────────────────
# 配置区：RSS 源
# ──────────────────────────────────────────────
RSS_FEEDS: list[dict] = [
    {
        "name": "arXiv cs.AI",
        "url": "https://rss.arxiv.org/rss/cs.AI",
        "category": "research",
    },
    {
        "name": "arXiv cs.CL",
        "url": "https://rss.arxiv.org/rss/cs.CL",
        "category": "research",
    },
    {
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss.xml",
        "category": "industry",
    },
    {
        "name": "Google DeepMind",
        "url": "https://deepmind.google/blog/rss.xml",
        "category": "industry",
    },
    {
        "name": "Anthropic",
        "url": "https://www.anthropic.com/rss.xml",
        "category": "industry",
    },
    {
        "name": "Hugging Face Blog",
        "url": "https://huggingface.co/blog/feed.xml",
        "category": "tools",
    },
    {
        "name": "The Batch (deeplearning.ai)",
        "url": "https://www.deeplearning.ai/the-batch/feed/",
        "category": "community",
    },
]

REPO_ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = REPO_ROOT / "journal"
LANDSCAPE_FILE = REPO_ROOT / "landscape" / "model-tracker.md"
TEMPLATE_FILE = JOURNAL_DIR / "_template.md"

TODAY = dt.date.today()


# ──────────────────────────────────────────────
# RSS 拉取
# ──────────────────────────────────────────────
def fetch_feed(feed_cfg: dict, since: dt.datetime) -> list[dict]:
    """拉取单个 RSS 源中比 since 新的条目。"""
    entries = []
    try:
        parsed = feedparser.parse(feed_cfg["url"])
        for entry in parsed.entries[:30]:  # 取最新 30 条
            published = None
            for attr in ("published_parsed", "updated_parsed"):
                ts = getattr(entry, attr, None)
                if ts:
                    published = dt.datetime(*ts[:6])
                    break
            if published and published < since:
                continue
            entries.append(
                {
                    "title": getattr(entry, "title", "Untitled"),
                    "link": getattr(entry, "link", ""),
                    "summary": getattr(entry, "summary", "")[:300],
                    "source": feed_cfg["name"],
                    "category": feed_cfg["category"],
                }
            )
    except Exception as e:
        print(f"  [WARN] 拉取 {feed_cfg['name']} 失败: {e}")
    return entries


def fetch_all_feeds(since: dt.datetime) -> list[dict]:
    """拉取所有 RSS 源。"""
    all_entries: list[dict] = []
    for feed in RSS_FEEDS:
        print(f"  拉取 {feed['name']} ...")
        entries = fetch_feed(feed, since)
        all_entries.extend(entries)
        print(f"    → {len(entries)} 条新条目")
    return all_entries


# ──────────────────────────────────────────────
# LLM 摘要 (可选)
# ──────────────────────────────────────────────
def summarize_with_llm(entries: list[dict]) -> str:
    """调用 OpenAI 生成中文摘要日报。"""
    from openai import OpenAI

    client = OpenAI()

    bullet_list = "\n".join(
        f"- [{e['source']}] {e['title']}: {e['summary'][:150]}... ({e['link']})"
        for e in entries[:50]  # 限制条目数量避免 token 过多
    )

    prompt = textwrap.dedent(f"""\
        你是一个 AI 行业动态整理助手。请将以下今日 RSS 条目整理成一份中文日报。

        分为以下四个板块，每个板块列 3-5 条最重要的条目（如该板块无内容则写"（暂无）"）：
        1. 模型发布 — 新模型发布或重大更新
        2. 研究论文 — 值得关注的新论文
        3. 行业动态 — 公司动态、融资、产品发布
        4. 工具更新 — 开发工具、框架、平台的更新

        每条格式: `- **标题**: 一句话中文摘要 — [链接](url)`

        今日条目:
        {bullet_list}

        请只输出日报正文（不要输出 frontmatter 或标题）。
    """)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def format_entries_simple(entries: list[dict]) -> str:
    """无 LLM 时的简单格式化。"""
    by_cat = {"research": [], "industry": [], "tools": [], "community": []}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)

    cat_names = {
        "research": "研究论文",
        "industry": "行业动态",
        "tools": "工具更新",
        "community": "社区资讯",
    }

    sections = []
    for cat_key, cat_label in cat_names.items():
        items = by_cat.get(cat_key, [])[:8]
        section = f"## {cat_label}\n\n"
        if items:
            for e in items:
                section += f"- **{e['title']}** ({e['source']}) — [{e['link']}]({e['link']})\n"
        else:
            section += "- （暂无）\n"
        sections.append(section)

    return "\n\n".join(sections)


# ──────────────────────────────────────────────
# 写入 journal
# ──────────────────────────────────────────────
def write_journal(date: dt.date, body: str) -> Path:
    """写入 journal/YYYY/MM/DD.md。"""
    journal_path = JOURNAL_DIR / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}.md"
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    content = textwrap.dedent(f"""\
        ---
        date: {date.isoformat()}
        type: daily
        ---

        # {date.isoformat()} AI 日报

    """)
    content += body + "\n"

    journal_path.write_text(content, encoding="utf-8")
    print(f"✓ 日报已写入 {journal_path}")
    return journal_path


# ──────────────────────────────────────────────
# 更新 model-tracker (追加)
# ──────────────────────────────────────────────
def update_model_tracker(date: dt.date, entries: list[dict]) -> None:
    """如果有模型发布相关条目，追加到 model-tracker.md 的 '最近更新' 区。"""
    model_keywords = [
        "release", "launch", "model", "llm", "gpt", "claude", "gemini",
        "llama", "mistral", "deepseek", "qwen", "gemma", "phi",
    ]

    model_entries = [
        e for e in entries
        if any(kw in (e["title"] + e["summary"]).lower() for kw in model_keywords)
    ]

    if not model_entries:
        return

    if not LANDSCAPE_FILE.exists():
        return

    existing = LANDSCAPE_FILE.read_text(encoding="utf-8")

    # 在 "## 最近更新" 下方追加
    new_lines = [f"\n### {date.isoformat()}\n"]
    for e in model_entries[:5]:
        new_lines.append(f"- **{e['title']}** ({e['source']}) — [{e['link']}]({e['link']})")
    new_lines.append("")

    insert_text = "\n".join(new_lines)

    marker = "<!-- 日更脚本会在这里追加新条目，格式如下 -->"
    if marker in existing:
        updated = existing.replace(marker, marker + "\n" + insert_text)
    else:
        # fallback: 追加到文件末尾
        updated = existing.rstrip() + "\n\n" + insert_text

    LANDSCAPE_FILE.write_text(updated, encoding="utf-8")

    # 更新 frontmatter 中的 updated 日期
    updated = re.sub(
        r"^(updated:) .+$",
        f"\\1 {date.isoformat()}",
        updated,
        count=1,
        flags=re.MULTILINE,
    )
    LANDSCAPE_FILE.write_text(updated, encoding="utf-8")

    print(f"✓ model-tracker 已追加 {len(model_entries)} 条模型动态")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI 知识库每日更新脚本")
    parser.add_argument("--no-llm", action="store_true", help="不使用 LLM 生成摘要")
    parser.add_argument("--hours", type=int, default=24, help="拉取过去 N 小时的条目 (默认 24)")
    args = parser.parse_args()

    since = dt.datetime.now() - dt.timedelta(hours=args.hours)
    print(f"[{TODAY}] 拉取过去 {args.hours} 小时的 AI 动态...\n")

    entries = fetch_all_feeds(since)
    print(f"\n共拉取 {len(entries)} 条新条目\n")

    if not entries:
        print("无新条目，跳过日报生成。")
        return

    # 生成日报正文
    use_llm = not args.no_llm and os.environ.get("OPENAI_API_KEY")
    if use_llm:
        print("调用 LLM 生成摘要...")
        body = summarize_with_llm(entries)
    else:
        if not args.no_llm:
            print("[INFO] 未设置 OPENAI_API_KEY，使用简单格式化")
        body = format_entries_simple(entries)

    # 写入 journal
    write_journal(TODAY, body)

    # 更新 model-tracker
    update_model_tracker(TODAY, entries)


if __name__ == "__main__":
    main()
