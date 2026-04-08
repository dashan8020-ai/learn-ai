---
title: "序列建模架构 (Sequence Modeling Architectures)"
description: "Transformer、Mamba/SSM 及混合架构的深度技术解析——从注意力机制到状态空间模型，追踪序列建模范式的演进。"
created: 2026-04-08
updated: 2026-04-08
tags: [transformer, mamba, ssm, attention, architecture, rope, flash-attention, state-space-model]
---

# 序列建模架构 (Sequence Modeling Architectures)

> 序列建模是语言模型的底层基础设施。本文档聚焦两大范式——**Transformer（注意力）** 与 **Mamba/SSM（状态空间）**——的架构原理、设计权衡和工程实现，以及二者的混合探索。

---

## 1. Transformer

### 1.1 架构原理

Transformer 由 Vaswani 等人在 2017 年提出[^vaswani-2017]，用**纯注意力机制**替代了 RNN 的循环结构。原始设计是 Encoder-Decoder 架构（用于机器翻译），但后续演化出三个分支：

| 变体 | 代表模型 | 适用任务 | 关键差异 |
|------|----------|----------|----------|
| **Encoder-only** | BERT, RoBERTa | 分类、NER、语义理解 | 双向注意力，看到完整上下文 |
| **Decoder-only** | GPT 系列, Llama, Claude | 文本生成、对话、推理 | 因果掩码，只看到左侧 token |
| **Encoder-Decoder** | T5, BART, UL2 | Seq2Seq（翻译、摘要） | 编码器双向 + 解码器因果 |

**为什么 Decoder-only 成为 LLM 的主流？** 这不是偶然的——有深层的 scaling 原因：

1. **统一的训练目标**：Decoder-only 只做 next-token prediction，目标函数简单，不需要区分编码/解码阶段，scaling 行为更可预测
2. **参数效率**：Encoder-Decoder 要分两组参数，总参数量相同时 Decoder-only 的生成能力更强
3. **涌现能力的载体**：in-context learning、chain-of-thought 等涌现能力天然适配自回归生成
4. **工程简洁**：一个统一的 forward pass，推理优化（KV cache、投机解码）更容易实现

### 1.2 自注意力机制

自注意力是 Transformer 的核心计算单元。对输入序列中的每个 token，计算它与所有其他 token 的关联权重：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- `Q`（Query）、`K`（Key）、`V`（Value）：输入向量经过三个不同的线性投影
- `√d_k`（缩放因子）：防止点积过大导致 softmax 梯度消失

**为什么要除以 √d_k？** 当 `d_k` 较大时，`QK^T` 的方差约为 `d_k`，除以 `√d_k` 将方差归一化到 1，使 softmax 不会退化成 one-hot（梯度几乎为零）。这个看似简单的 trick 对训练稳定性至关重要。

**Multi-Head Attention (MHA)** 将 Q/K/V 拆分成 `h` 个头并行计算，每个头学习不同的注意力模式（如语法关系、共指消解、位置邻近性等），最后拼接：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 1.3 注意力变体：效率与质量的权衡

标准 MHA 的问题是推理时 KV cache 的内存占用随头数线性增长。这催生了一系列 KV 头共享方案：

| 方案 | KV 头数 | 代表模型 | 核心权衡 |
|------|---------|----------|----------|
| **MHA** | 等于 Q 头数 | GPT-3 | 质量最优，内存最大 |
| **MQA** | 1 | PaLM, Falcon | 内存最小，质量有损 |
| **GQA** | Q 头数 / G | Llama 2/3, Mistral | 分组折中，质量接近 MHA |

**GQA 为什么成为事实标准？** Llama 2 的实验表明 GQA-8（8 组 KV 头）在质量上几乎无损于完整 MHA，但推理吞吐提升 ~1.5x。这是一个 Pareto 最优点——进一步减少 KV 头收益递减，而质量损失加速[^llama2-2023]。

#### Multi-head Latent Attention (MLA)

DeepSeek-V2 提出的 MLA 走了一条不同于 GQA 的路径[^deepseek-v2-2024]：不是共享 KV 头，而是**将 KV 投影到低秩潜空间**再缓存。

```
标准 GQA:  缓存 [K_group, V_group] → 大小 ∝ n_kv_heads × d_head
MLA:       缓存 compressed_kv       → 大小 ∝ d_compress (远小于 n_kv_heads × d_head)
           推理时: compressed_kv → 上投影还原 K, V
```

MLA 的 KV cache 压缩比可达 **93.3%**（DeepSeek-V2 数据），同时保持了完整 MHA 的表达能力。这是因为低秩投影保留了 KV 的主要信息，而 GQA 的硬共享则丢弃了头间差异。

代价是推理时需要额外的上投影计算，但这被 KV cache 内存节省所带来的更大 batch size 抵消。DeepSeek-V3/R1 延续了这一设计。

### 1.4 Flash Attention：IO 感知的算法革新

Flash Attention[^flash-attention-2022] 不改变注意力的数学结果，而是重新组织计算顺序以减少 GPU HBM（高带宽内存）访问：

**核心问题**：标准注意力实现需要先在 HBM 中实体化 `n×n` 的注意力矩阵 `S = QK^T`，然后读取 `S` 做 softmax，再与 `V` 相乘。这三次 HBM 读写是瓶颈——SRAM 速度是 HBM 的 ~10x，但容量小得多。

**解法（Tiling + Online Softmax）**：

1. 将 Q/K/V 分成小块（tile），每块可以放进 SRAM
2. 在 SRAM 内完成 `QK^T → softmax → ×V` 的全部计算
3. 使用 online softmax（Milakov & Gimelshein, 2018）在不看到完整行的情况下增量计算正确的 softmax
4. 永远不把 `n×n` 注意力矩阵写入 HBM

**效果**：2-4x 加速，内存从 O(n²) 降到 O(n)，且结果**精确等价**（非近似）。

Flash Attention 2 进一步优化了并行性（在序列长度维度并行而非 batch 维度），Flash Attention 3 利用了 Hopper GPU 的异步特性（TMA + WGMMA）。

**为什么 Flash Attention 如此重要？** 它证明了一个原则：**在现代硬件上，算法的 IO 复杂度往往比计算复杂度更重要**。标准注意力的 FLOPs 完全相同，但 IO 模式决定了实际速度。这一洞察影响了后续几乎所有高效架构的设计。

### 1.5 位置编码的演进

Transformer 本身是**置换不变**的——打乱输入顺序不影响输出。位置编码注入序列顺序信息：

| 方案 | 机制 | 外推性 | 代表模型 |
|------|------|--------|----------|
| **正弦位置编码** | 固定的 sin/cos 函数 | 差 | 原始 Transformer |
| **可学习位置编码** | 训练出的 embedding | 无 | GPT-2, BERT |
| **RoPE** | 旋转矩阵编码相对位置 | 中（需配合外推） | Llama, Qwen, Mistral |
| **ALiBi** | 注意力分数加线性偏置 | 好 | BLOOM, MPT |

#### RoPE 的技术本质

RoPE（Rotary Position Embedding）[^rope-2021] 的核心思想：将位置信息编码为**旋转角度**，使得两个 token 的注意力分数只取决于它们的**相对距离**。

```
q_m = R(mθ) · q    # 位置 m 的 query 乘以旋转矩阵 R(mθ)
k_n = R(nθ) · k    # 位置 n 的 key 乘以旋转矩阵 R(nθ)
q_m · k_n = q^T R((m-n)θ) k    # 点积只依赖相对位置 (m-n)
```

每个维度对 `(d_{2i}, d_{2i+1})` 以频率 `θ_i = 10000^{-2i/d}` 旋转。低频维度编码远距离关系，高频维度编码近距离关系——类似傅里叶分解。

**长度外推问题**：RoPE 在训练长度内效果优异，但超出训练长度后角度外推会产生分布外的注意力模式。解决方案：

- **NTK-aware Scaling**：调大基础频率 `θ`，等效于"放大旋转盘"，低频维度不受影响、高频维度被压缩
- **YaRN**：NTK 的改进版，对不同频率维度施加不同的缩放策略
- **持续预训练**：在目标长度的数据上继续训练（最可靠，但最昂贵）

### 1.6 归一化：训练稳定性的关键

归一化层的位置和类型对深度 Transformer 的训练稳定性影响巨大：

| 方案 | 位置 | 特点 |
|------|------|------|
| **Post-LN** | Attention/FFN 后 | 原始 Transformer；深度增加时梯度不稳定 |
| **Pre-LN** | Attention/FFN 前 | 训练稳定但最终层输出无归一化 |
| **Sandwich-LN** | 前后各一个 | 同时保证稳定性和输出归一化 |

**为什么 Pre-LN 更稳定？** 在 Post-LN 中，残差路径上的激活值随深度累积增长（因为归一化在残差相加之后），导致深层梯度爆炸。Pre-LN 在进入注意力/FFN 之前归一化，残差路径保持"干净"，梯度可以畅通地反向传播。

**RMSNorm 取代 LayerNorm**：LayerNorm 计算均值和方差两个统计量；RMSNorm 只计算均方根（root mean square），省去了均值计算。实验表明性能几乎无损，但速度更快。Llama 系列率先采用 RMSNorm，现已成为标准。

### 1.7 前馈网络与激活函数

Transformer 每层包含一个前馈网络（FFN），负责非线性变换和知识存储。研究表明 FFN 是模型记忆事实知识的主要载体（key-value memory 假说[^geva-2021]）。

**激活函数演进**：

```
ReLU: max(0, x)                    — 简单但有"死神经元"问题
GELU: x · Φ(x)                    — 平滑版 ReLU，GPT-2/BERT 采用
SwiGLU: Swish(xW₁) ⊙ (xW₂)       — 门控线性单元，现代 LLM 标配
```

**SwiGLU 为什么是当前最优？** GLU（Gated Linear Unit）引入了一个门控路径，让网络可以选择性地让信息通过[^shazeer-2020]。Shazeer 2020 年的实验表明 SwiGLU 在相同参数量下 consistently outperform GELU/ReLU。直觉上，门控机制让 FFN 获得了类似注意力的"选择性"——不是所有输入维度都同等对待。

### 1.8 MoE：稀疏激活的参数扩展

Mixture of Experts 不是独立的架构，而是 Transformer FFN 层的一种替换策略：

```
标准 FFN:   x → FFN(x)
MoE FFN:    x → Router(x) → 选择 Top-K Expert → Σ(gate_i · Expert_i(x))
```

| 模型 | 总参数 | 活跃参数 | Expert 数 | Router 策略 |
|------|--------|----------|-----------|-------------|
| Mixtral 8x7B | 46.7B | 12.9B | 8, Top-2 | Token-choice |
| DeepSeek-V3 | 671B | 37B | 256, Top-8 | 辅助 loss-free 均衡 |
| Qwen2.5-MoE | 14.3B | 2.7B | 60, Top-4+4 shared | Fine-grained |

**MoE 的关键工程挑战**：

1. **负载均衡**：如果 Router 总选同几个 Expert，其他 Expert 白训练。传统方案加辅助 loss 强制均衡，但这个 loss 会干扰主任务。DeepSeek-V3 提出 **loss-free 均衡**——用动态偏置调节 Router 分数，不引入额外 loss
2. **Expert 并行**：256 个 Expert 不可能放在一张卡上，需要跨设备调度。通信开销是瓶颈
3. **Token 丢弃**：当某个 Expert 过载时需要丢弃 token，影响质量

---

## 2. 状态空间模型 (SSM) 与 Mamba

### 2.1 为什么需要 Transformer 的替代方案？

Transformer 的核心瓶颈是**注意力的二次复杂度**：

| 操作 | 训练复杂度 | 推理复杂度（生成第 t 个 token） |
|------|-----------|-------------------------------|
| 自注意力 | O(n²d) | O(td)（需看所有历史，但 KV cache 帮助） |
| KV cache 内存 | — | O(n · d · layers)，随上下文线性增长 |
| FFN | O(nd²) | O(d²)（与序列长度无关） |

当上下文长度达到 128K-2M 时，KV cache 内存成为实际部署的主要瓶颈。例如，Llama-70B 在 128K 上下文时，KV cache 需要 ~40GB——可能比模型参数本身还大。

这催生了对**线性复杂度序列建模**的需求。

### 2.2 从 S4 到 Mamba：SSM 的演进

状态空间模型（State Space Model）将序列建模形式化为连续动力系统的离散化：

```
连续形式:
  h'(t) = Ah(t) + Bx(t)    # 状态转移
  y(t)  = Ch(t) + Dx(t)    # 输出

离散化 (零阶保持):
  h_t = Ā h_{t-1} + B̄ x_t
  y_t = C h_t + D x_t
```

其中 `h` 是隐状态，`A` 是状态转移矩阵（编码了"记忆"的结构），`B`/`C` 是输入/输出投影。

#### S4（Structured State Spaces for Sequence Modeling, 2021）[^s4-2021]

S4 的关键突破是发现 `A` 矩阵的初始化方式决定了一切：

- 使用 **HiPPO（High-order Polynomial Projection Operator）** 矩阵初始化 `A`，使隐状态能够最优地压缩输入历史
- HiPPO 的数学含义：将输入信号投影到 Legendre 多项式基上，等效于对历史信号的最优低秩近似

S4 在长程建模基准（Long Range Arena, Path-X）上大幅超越 Transformer，但在语言建模上表现不佳。原因：**A, B, C 是固定的（输入无关）**，模型无法根据当前输入选择性地记忆或遗忘。

#### H3（Hungry Hungry Hippos, 2022）[^h3-2022]

H3 在 S4 基础上加入了类似注意力的门控机制和乘法交互，缩小了与 Transformer 在语言任务上的差距。更重要的是，H3 揭示了 SSM 与注意力之间的数学联系——SSM 可以视为一种**结构化的线性注意力**。

### 2.3 Mamba：选择性状态空间模型

Mamba[^mamba-2023] 是 SSM 范式的决定性突破，首次在语言建模上匹配 Transformer 质量，同时保持线性复杂度。

**Mamba 的核心创新：选择性机制（Selection Mechanism）**

S4 的 A, B, C 矩阵是固定参数；Mamba 让 **B, C 和时间步长 Δ 成为输入的函数**：

```
S4:    B, C, Δ = 固定参数
Mamba: B = f_B(x_t), C = f_C(x_t), Δ = f_Δ(x_t)    # 输入依赖
```

这个看似简单的改变有深刻的含义：

1. **选择性记忆**：模型可以根据当前输入决定记多少（大 Δ → 快速吸收新信息，遗忘旧状态）或忘多少（小 Δ → 保持旧记忆，忽略当前输入）
2. **等效于门控 RNN**：选择性 SSM 在概念上等价于一个数据依赖的门控 RNN，但计算方式完全不同
3. **内容感知推理**：Transformer 的注意力天然是内容感知的（token 间的相似度决定权重）；选择性机制让 SSM 也获得了这种能力

**但选择性破坏了高效计算**：S4 之所以高效，是因为固定的 A, B, C 可以预计算卷积核，用 FFT 做 O(n log n) 的序列并行。参数变成输入依赖后，卷积不再适用。

**硬件感知的并行扫描算法**：

Mamba 的工程创新在于将选择性 SSM 实现为**并行前缀和（parallel scan）**，利用结合律在 GPU 上高效并行：

```
朴素循环: h_t = Ā_t h_{t-1} + B̄_t x_t    # O(n) 顺序计算
并行扫描: 将 (Ā_t, B̄_t x_t) 视为半群元素 → 前缀和 → O(log n) 并行深度
```

配合 Flash Attention 式的 kernel fusion（在 SRAM 中完成所有中间计算，避免 HBM 读写），Mamba 在 A100 上的实际速度是同参数量 Transformer 的 **3-5x**（长序列时）。

**Mamba 架构细节**：

```
输入 x
 ├── 线性投影 → 扩展到 2E 维度（E = expand factor, 通常 2）
 ├── 分支 1: Conv1d → SiLU → 选择性 SSM → ...
 ├── 分支 2: SiLU (门控路径) → ...
 └── 两分支逐元素相乘 → 线性投影回 D 维度 → 输出
```

注意 Mamba **没有注意力层、没有 MLP 层**——整个 block 就是上面这个结构重复 N 次。比 Transformer 的 Attention+FFN 更简洁。

### 2.4 Mamba-2：结构化状态空间对偶性

Mamba-2[^mamba2-2024] 从理论上统一了 SSM 和注意力机制：

**SSD（Structured State Space Duality）框架**：

证明了选择性 SSM 等价于一种**半可分矩阵（semiseparable matrix）** 形式的结构化注意力。具体地：

```
SSM 视角:  h_t = A_t h_{t-1} + B_t x_t, y_t = C_t h_t
注意力视角: y = M ⊙ (QK^T) V
           其中 M 是因果掩码 × 衰减矩阵
```

两种计算给出**数学上完全等价**的结果，但计算复杂度不同：

- **SSM 模式**（循环）：O(n) 时间，O(1) 推理步骤内存 → 适合推理
- **注意力模式**（矩阵乘法）：O(n²) 时间，高并行度 → 适合训练

Mamba-2 的实际做法是**分块处理**：将序列分成长度 T 的块，块内用矩阵乘法（利用 GPU tensor core），块间用循环传递状态。这是两种模式的最优混合。

**性能提升**：Mamba-2 比 Mamba-1 训练速度快 2-8x，同时在 scaling 行为上更接近 Transformer++（带 GQA、SwiGLU 等优化的 Transformer）。

### 2.5 SSM vs Transformer：能力边界

SSM 并非万能。关键实验发现：

| 能力 | Transformer | Mamba | 原因 |
|------|-------------|-------|------|
| **In-context learning** | 强 | 弱 | 注意力可以直接"查表"（KV 精确匹配）；SSM 必须将信息压缩进固定大小的隐状态 |
| **精确回忆** | 强（KV cache） | 弱 | 隐状态大小固定，信息量有上限 |
| **长程依赖** | 受窗口限制 | 理论上无限 | SSM 的隐状态可以携带任意远的信息（但有损） |
| **推理效率** | KV cache 随长度增长 | O(1) 状态 | SSM 不需要缓存历史 token |
| **归纳推理** | 弱（但可通过 CoT 缓解） | 更弱 | 两者都不擅长，但 Transformer 的精确回忆能力有助于 CoT |

**核心权衡**：Transformer 用 O(n) 内存（KV cache）换取精确的信息检索；SSM 用 O(1) 内存实现高效推理，但信息必须经过有损压缩。这是信息论意义上的根本矛盾。

---

## 3. 替代架构

### 3.1 RWKV（Receptance Weighted Key Value）

RWKV[^rwkv-2023] 试图将 Transformer 的 attention "线性化"并转换为 RNN 形式：

- **训练时**：按 Transformer 风格并行（矩阵运算）
- **推理时**：按 RNN 风格循环（O(1) 步骤复杂度）

RWKV 的关键设计是 **WKV（Weighted Key-Value）** 操作，类似于 attention 但用指数衰减替代 softmax：

```
wkv_t = Σ_{i=1}^{t-1} e^{-(t-1-i)w + k_i} · v_i + e^{u+k_t} · v_t
```

其中 `w` 是可学习的时间衰减参数。这个公式可以递推计算，因此推理时是 O(1)。

**RWKV 的局限**：时间衰减是单调递减的，无法像注意力那样"跳跃式"地关注远处的特定 token。RWKV-6/7 通过数据依赖的衰减和门控机制持续改进，但在精确回忆任务上仍不及 Transformer。

当前 RWKV 发展到 v7 (Goose)，社区活跃，最大模型到 14B 规格，是纯开源社区驱动的唯一竞争性 LLM 架构。

### 3.2 xLSTM（Extended LSTM）

xLSTM[^xlstm-2024] 由 LSTM 的发明者 Sepp Hochreiter 提出，本质是"如果用现代技术重新设计 LSTM 会怎样"：

- **sLSTM（scalar LSTM）**：引入指数门控，缓解梯度消失
- **mLSTM（matrix LSTM）**：将标量记忆扩展为矩阵记忆，容量大幅提升。mLSTM 的更新规则与线性注意力 + 衰减非常相似，可以并行训练

xLSTM 在中等规模实验中表现不错（400M-1.3B），但缺乏大规模 scaling 的验证。

### 3.3 RetNet（Retentive Network）

微软提出的 RetNet[^retnet-2023] 引入 **retention 机制**——一种带指数衰减的注意力变体：

```
Retention(Q, K, V) = (QK^T ⊙ D) V    其中 D_{ij} = γ^{i-j} (因果衰减掩码)
```

支持三种等价的计算模式：
1. **并行模式**：类注意力的矩阵乘法 → 适合训练
2. **循环模式**：类 RNN → 适合推理
3. **分块模式**：块内并行 + 块间循环 → 平衡训练和推理

这个三模式等价性后来被 Mamba-2 的 SSD 框架所吸收和泛化。

---

## 4. 混合架构：融合两个范式

纯 SSM 在精确回忆和 in-context learning 上弱于 Transformer，纯 Transformer 在长序列推理效率上有瓶颈。混合架构尝试两全其美。

### 4.1 Jamba（AI21 Labs）

Jamba[^jamba-2024] 是第一个大规模部署的 Transformer-Mamba 混合模型：

```
Jamba 架构 (52B 总参 / 12B 活跃参):
  Layer 1-4:   Mamba block
  Layer 5:     Transformer attention block
  Layer 6-9:   Mamba block
  Layer 10:    Transformer attention block + MoE
  ...重复...
```

**设计原理**：大部分层用 Mamba（高效处理序列），每隔几层插入一个 Transformer 注意力层（提供精确的信息检索能力）。MoE 进一步扩大参数容量。

**效果**：256K 上下文窗口；相比同参数量的纯 Transformer，推理吞吐提升 3x，质量持平。

### 4.2 Zamba（Zyphra）

Zamba 的独特设计：**所有 Mamba 层之间共享同一个注意力层**：

```
Mamba → 共享 Attention → Mamba → 共享 Attention → Mamba → ...
```

这进一步压缩了参数量（注意力层只有一份权重），特别适合端侧部署。Zamba-2 在 2.7B 参数量级上达到了可观的质量。

### 4.3 其他混合探索

| 模型 | 组织 | 混合方式 |
|------|------|----------|
| **Griffin** | Google DeepMind | 循环层 (RLKV) + 局部注意力 |
| **RecurrentGemma** | Google | 基于 Griffin 的开源模型 |
| **StripedHyena** | Together AI | Hyena (长卷积) + 注意力交替 |
| **Samba** | Microsoft | Mamba + Sliding Window Attention |

**混合比例是关键**：研究表明注意力层的比例只需 ~15-25% 即可恢复大部分精确回忆能力，更多的注意力层对推理效率的损害大于质量收益。

---

## 5. 架构选择的工程考量

### 决策矩阵

| 场景 | 推荐架构 | 理由 |
|------|----------|------|
| 通用 LLM（追求最强质量） | Transformer + GQA/MLA | 生态最成熟，scaling 行为最可预测 |
| 超长上下文（>256K） | 混合 Mamba-Transformer | KV cache 是纯 Transformer 的硬瓶颈 |
| 端侧部署 | SSM/混合 + 量化 | O(1) 推理内存，不受上下文长度限制 |
| 高吞吐服务 | 依场景而定 | 短 prompt 选 Transformer（prefill 并行度高）；长 prompt 选混合 |
| 实时流式处理 | SSM/RWKV | 天然的流式推理，无需 KV cache 管理 |

### 2026 年现状

截至 2026 年 4 月，**Transformer 仍是绝对主流**。所有前沿闭源模型（GPT-5.3, Claude Opus 4.6, Gemini 3）都基于 Transformer（可能有内部改良，但核心是注意力机制）。SSM/Mamba 在开源社区和特定场景（长序列、端侧）中持续推进，但尚未在 >70B 规模上证明可以替代 Transformer。

混合架构是最有前景的方向——不是"Transformer 或 SSM"的二选一，而是在**同一模型内按层分配**注意力和循环计算的比例。

---

## 参考资料

[^vaswani-2017]: Vaswani et al. *Attention Is All You Need*. 2017. https://arxiv.org/abs/1706.03762
[^llama2-2023]: Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*. 2023. https://arxiv.org/abs/2307.09288
[^deepseek-v2-2024]: DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. 2024. https://arxiv.org/abs/2405.04434
[^flash-attention-2022]: Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. 2022. https://arxiv.org/abs/2205.14135
[^rope-2021]: Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021. https://arxiv.org/abs/2104.09864
[^shazeer-2020]: Shazeer. *GLU Variants Improve Transformer*. 2020. https://arxiv.org/abs/2002.05202
[^geva-2021]: Geva et al. *Transformer Feed-Forward Layers Are Key-Value Memories*. 2021. https://arxiv.org/abs/2012.14913
[^s4-2021]: Gu et al. *Efficiently Modeling Long Sequences with Structured State Spaces*. 2021. https://arxiv.org/abs/2111.00396
[^h3-2022]: Fu et al. *Hungry Hungry Hippos: Towards Language Modeling with State Space Models*. 2022. https://arxiv.org/abs/2212.14052
[^mamba-2023]: Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. 2023. https://arxiv.org/abs/2312.00752
[^mamba2-2024]: Dao & Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. 2024. https://arxiv.org/abs/2405.21060
[^rwkv-2023]: Peng et al. *RWKV: Reinventing RNNs for the Transformer Era*. 2023. https://arxiv.org/abs/2305.13048
[^xlstm-2024]: Beck et al. *xLSTM: Extended Long Short-Term Memory*. 2024. https://arxiv.org/abs/2405.04517
[^retnet-2023]: Sun et al. *Retentive Network: A Successor to Transformer for Large Language Models*. 2023. https://arxiv.org/abs/2307.08621
[^jamba-2024]: Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model*. 2024. https://arxiv.org/abs/2403.19887
