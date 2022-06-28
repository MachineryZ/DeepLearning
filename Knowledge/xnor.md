# XNOR 

XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks

PDF:
https://arxiv.org/abs/1603.05279.pdf

Code: 
https://allenai.org/plato/xnornet



<div align=center><img src="../Files/xnor1.jpeg" width=90%></div>

本文本质上提出了两种 binary CNN 的形式来压缩、加速 NN 的 inference 的速度：

- Binary-weights，所有网络卷积层参数为 binary 形式
- XNOR-Networks，所有网络卷积层参数 和 网络输入 都为 binary 形式


- Binary-Weight-Networks
    - 用一个 binary 的 conv filter 再加上一个 scaler 参数 $\alpha$ 来近似原始的卷积核，$\oplus$ 是纯卷积，没有任何加操作。 

$$
I * B \approx (I \oplus \alpha B)
$$

- Estimating Binary Weights，假设 $W,B\in \mathcal{R}^n,n=c\times w\times h$，我们要解一个优化问题
$$
J(B,\alpha) = ||W - \alpha B||^2 \\
\alpha^*, B^* = \arg\min_{\alpha,B}J(B,\alpha)
$$

上述优化问题，可以转化为：
$$
B^* = \argmax_{B}\{W^TB\}, \ \ s.t.\ \ B\in\{+1,-1\}^n \\
\Rightarrow B^*=\text{sign}(W), \alpha^*=W^T B^*/n \\
\Rightarrow \alpha^*=W^TB^*/n=W^T sign(W)/n=\sum |W_i|/n=||W||_{l1}/n
$$

也就是说，根据上述的定义，给定任意的卷积核参数 $W$，我们就能有一个近似的最优 binary 卷积核 $B^*$ 和 他的 scaler $\alpha^*$

- Training Binary-Weights-Networks：在说如何训练之前，我们需要知道符号函数的求导问题。符号函数，在 0 点无法定义导数，其他地方都是 0，所以我们在 bp 的过程中需要对符号函数进行松弛求解。
$$
q = sign(r)
$$
假设 q 的梯度为，$C$ 为损失函数，网络 output 和 groundtruth 计算得到的损失函数
$$
g_q = \partial C / \partial q
$$
那么损失函数 $C$ 对 r 的求导公式如下：

$$
g_r = \partial C/ \partial q · \partial q/\partial r=g_q 1_{|r|\leq 1}
$$
其中，$ 1_{|r|\leq 1}$ 的计算公式为 Htanh，相当于强行将 $\{-1,+1\}$ 区间加入了可导的导数
$$
Htanh(x) = Clip(x, -1,1) = \max(-1,\min(1,x))
$$


<div align=center><img src="../Files/xnor2.jpeg" width=90%></div>






