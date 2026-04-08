"""
scrapers — 每个网站一个独立爬虫模块。

每个模块须暴露:
    NAME: str       — 显示名称
    SLUG: str       — 输出文件名（不含 .md）
    CATEGORY: str   — 分类 key（industry / community / …）

    def scrape() -> list[dict]
        返回条目列表，每个 dict 包含:
            title, link, summary, source, slug, cat
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _discover() -> dict[str, "ModuleType"]:
    """自动发现本包下所有爬虫模块，返回 {slug: module} 映射。"""
    modules: dict[str, ModuleType] = {}
    package_path = __path__  # type: ignore[name-defined]
    for info in pkgutil.iter_modules(package_path):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        if callable(getattr(mod, "scrape", None)):
            slug = getattr(mod, "SLUG", info.name)
            modules[slug] = mod
    return modules


def get(slug: str) -> "ModuleType | None":
    """按 slug 查找爬虫模块，找不到返回 None。"""
    return _discover().get(slug)


def run_scraper(cfg: dict) -> list[dict]:
    """根据 feeds.yaml 中的 scrape 条目调用对应爬虫。

    cfg 至少包含: name, slug, cat
    """
    slug = cfg["slug"]
    mod = get(slug)
    if mod is None:
        print(f"  [WARN] 未找到爬虫模块: scrapers/{slug}.py")
        return []
    try:
        return mod.scrape()
    except Exception as exc:
        print(f"[WARN] {exc}")
        return []
