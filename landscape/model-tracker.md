---
title: 模型动态跟踪
description: 追踪主流 AI 模型的最新发布、更新和关键变化。本文件由日更脚本自动维护。
created: 2026-04-07
updated: 2026-04-07
tags: [models, landscape, tracker]
---

# 模型动态跟踪

> 本文件追踪主流 AI 模型的发布和更新动态，分为闭源模型与开源模型两部分。由日更脚本自动维护，也可手动补充。

---

## 闭源模型

| 模型 | 厂商 | 发布时间 | 上下文窗口 | 亮点 |
|------|------|----------|------------|------|
| **GPT-5.3-Codex** | OpenAI | 2026.4 | 128K+ | 专为代码优化，支持按需定价 |
| **GPT-5.3 Instant** | OpenAI | 2026.3 | 128K+ | 更快速、更经济 |
| **Gemini 3.1 Flash** | Google | 2026.3 | 1M+ | 高速推理，自然语音 |
| **Gemini 3.1 Flash-Lite** | Google | 2026.3 | 1M+ | 大规模部署优化 |
| **Gemini 3.1 Pro** | Google | 2026.2 | 1M+ | 长上下文，复杂任务 |
| **Gemini 3 Deep Think** | Google | 2026.2 | - | 科学研究与数学推理 |
| **GPT-5** | OpenAI | 2025 | 128K+ | 多模态原生，强推理能力 |
| **Claude Opus 4** | Anthropic | 2025 | 200K | 最强推理，深度分析 |
| **Claude Sonnet 4** | Anthropic | 2025 | 200K | 性能与速度平衡 |
| **Claude Haiku 3.5** | Anthropic | 2025 | 200K | 快速响应，成本最低 |
| **o3** | OpenAI | 2025 | 128K | 推理模型，o1 后续 |
| **o1** | OpenAI | 2024.9 | 128K | 首个商业推理模型 |

## 开源模型

| 模型 | 厂商 | 参数量 | 发布时间 | 亮点 |
|------|------|--------|----------|------|
| **Gemma 4** | Google | 多规格 | 2026.4 | 开源高效率 |
| **DeepSeek-R1** | DeepSeek | 671B | 2025.1 | 开源推理模型标杆 |
| **DeepSeek-V3** | DeepSeek | 671B(37B active) | 2024 | MoE 架构，卓越性价比 |
| **Llama 4** | Meta | 多规格 | 2025 | 开源旗舰，社区生态最强 |
| **Llama 3.1** | Meta | 8B/70B/405B | 2024 | 广泛使用 |
| **Mistral Large** | Mistral | 未公开 | 2024 | 欧洲 AI 代表 |
| **Mixtral 8x7B** | Mistral | 46.7B(12.9B active) | 2024 | MoE 架构先驱 |
| **Qwen2.5** | 阿里 | 0.5B-72B | 2024 | 中文最强开源之一 |

## 最近更新

<!-- 日更脚本会在这里追加新条目，格式如下 -->
<!-- ### 2026-04-07 -->
<!-- - [模型名] 发布/更新：简要说明 -->

---

## 评估排行 (Chatbot Arena)

[Chatbot Arena](https://chat.lmsys.org/) 是目前最权威的 LLM 综合评测平台，基于人类盲评 ELO 排名。

| 基准 | 评估能力 |
|------|----------|
| **MMLU / MMLU-Pro** | 知识广度与深度 |
| **HumanEval / SWE-bench** | 代码能力 |
| **MATH / GSM8K** | 数学推理 |
| **GPQA** | 专家级科学知识 |
| **Chatbot Arena (ELO)** | 综合对话 |
| **FACTS** | 事实性 (Google 2025.12) |
