# State Space Model

https://jameschen.io/jekyll/update/2024/02/12/mamba.html#tldr

S4 (Efficiently Modeling Long Sequences with Structured State Spaces)

https://arxiv.org/pdf/2111.00396.pdf

Mamba (Mamba: Linear-Time Sequence Modeling with Selective State Spaces)

https://arxiv.org/pdf/2312.00752.pdf

Hippo (Hungry Hungry Hippos: Towards Language Modeling with State Space Models)

https://arxiv.org/pdf/2212.14052.pdf



状态空间和状态空间模型 stat space & state space model，迷宫为例子

- 映射输出序列 $x(t)$，比如迷宫中移动方向，在迷宫中的坐标
- 潜在状态表示 $h(t)$，比如距离出口距离和 x/y 坐标
- 导出预测输出序列 $y(t)$

SSM 的两个方程，状态方程和输出方程：
$$
h'(t) = Ah(t) + Bx(t) \\
y(t) = Ch(t) + Dx(t)
$$
状态方程理解为：下一个时刻的状态由当前状态和当前信息决定，输出方程同理。

---

SSM 到 S4 的三步升级，离散化 SSM、循环/卷积表示、基于 HiPPO 处理长序列。离散数据的连续化，基于零阶保持技术 zero-order hold technique，步长保持（也就是阶段性保持）
$$
\bar{A} = e^{\Delta A} \\
\bar{B} = (\Delta A)^{-1} (e^{\Delta A} - I )\Delta B
$$
从而得到离散的状态方程和输出方程
$$
h_k = \bar Ah_{k-1} + \bar B x_k \\
y_k = Ch_k
$$
对序列的递推公式变为：
$$
y_2 = Ch_2 \\
= C(\bar Ah_1+ \bar B x_2) \\
= C(\bar A(\bar Ah_0 + \bar Bx_1) + \bar B x_2) \\
= C \bar A^2 \bar B x_0 + C \bar A \bar B x_1 + C\bar B x_2
$$
这个公式和卷积公式特别相似，所以我们可以表示这个“过滤器”
$$
\bar K = (C \bar B, C\bar A \bar B,...,C\bar A^k\bar B)
$$
所以把一个时序的问题转移到 CNN 的运算模式上，推理用RNN 训练用CNN，可以非常高效的计算为递归或卷积，在序列长度上具有线性或近线性缩放

矩阵 A 的问题与其解决之道 HiPPO

如我们之前在循环表示中看到的那样，矩阵 A 捕获先前 previous 状态的信息来构建新状态，怎样以保留比较长的 memory 的方式创建矩阵 A？答案是可以使用 Hungry Hungry Hippo（Hippo的全称是High-order Polynomial Projection Operator，简称 H3）Hippo尝试将当前看到的所有输入信号压缩为系数向量，它使用矩阵 A 构建了一个可以很好捕捉最近 token 并衰减旧 token的矩阵：
$$
A_{nk} = 
\begin{cases}
&(2n+1)^{1/2}(2k+1)^{1/2} \text{, below diag}\\
&(n+1) \text{, diagnal}\\
&0 \text{, above diag}\\
\end{cases}
$$
SSM的问题：矩阵固定不变，无法针对输入做针对性推理，矩阵ABC始终相同。

---

mamba：序列数据一般都是离散的数据，比如 文本、图、DNA

- 但现实生活中还有很多连续的数据，比如音频、视频，对于音视频这种信号而言，其一个重要特点就是有极长的context window
- 而 transformer 长 context上往往会失败，或者注意力机制在有着超长上下文长度的任务上并不擅长（所以才有各种对注意力机制的改进，比如 flash attention，bigbird，trnasformerXL等等，即便如此一般也就32k的上下文长度，在面对100w的序列长度则无能为力）但是 S4 擅长这类人物
- 比如 ema 来说，其实是可以有 unbounded context 无线长度的 window，transformer和convolution因为都只有着有限的上下文窗口而不好计算































