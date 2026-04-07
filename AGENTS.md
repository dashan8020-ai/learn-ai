# AGENTS.md — 项目约定

## 项目概述

个人 AI 前沿知识库。知识文档用中文编写，代码和配置用英文。

## 目录约定

- `docs/foundations/` — 基础理论（稳定知识，低频更新）
- `docs/applied/` — 应用技术（工程方法，中频更新）
- `docs/research/` — 前沿研究（快速演进，高频更新）
- `landscape/` — 行业全景（模型跟踪，由日更脚本维护）
- `journal/YYYY/MM/DD.md` — 每日动态（自动生成）
- `resources.md` — 精选资源汇总
- `scripts/` — 自动化脚本

## Markdown 规范

- 所有知识文档必须有 YAML frontmatter（title, description, created, updated, tags）
- 日报用简化 frontmatter（date, type）
- 每次修改知识文档时更新 `updated` 字段
- 参考资料统一放在文档末尾的 `## 参考资料` 区

## 日更脚本

- 入口：`scripts/daily_update.py`
- 依赖管理：PEP 723 inline metadata，由 `uv run` 自动安装
- RSS 源配置在脚本顶部的 `RSS_FEEDS` 列表中
- GitHub Actions：`.github/workflows/daily-update.yml`，每天 UTC 00:00 运行

## 构建 & 验证

- 无构建步骤，纯 Markdown 仓库
- 日更脚本测试：`uv run scripts/daily_update.py --no-llm --hours 1`
