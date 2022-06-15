# SVM

SVM（支持向量机）是一种二分类模型，他将实例的特征向量映射为空间中的一些点，SVM 的目的就是想要画出一些线，来“最好的”区分这些点。SVM 适合中小型数据样本、非线性、高纬度的分类问题。

1. 能够画出多少条线对样本点进行1区分？
线可以有无数条可以画，区别就在于效果好不好。

2. 为什么叫做“超平面”
因为样本的特征维度可能是高维的，高维空间中的“线”，“面”一般称为超平面

3. 划线的标准是什么？
线的标准是，能否将同一类的点分割在线的同一侧，不同类别的点分割在线的同一侧。

4. 间隔（margin）是什么？
对于任意一个超平面，其两侧数据点，都有距离它有一个最小距离（垂直距离），这个两个最小距离相加就是间隔。

对于一组数据，每个数据点可以用 $x_i \in \mathcal{R^{d}}$ 来表示，超平面方程为 $w^Tx+b=0$，两类的类别信息分别用 $y_{+} = 1，y_{-}=-1$ 来表示。那么 SVM 的优化问题可以写为：

$$
\begin{aligned}
T = {(x_1, y_1),...,(x_n,y_n)}, \quad x_i\in\mathcal{R^d},\quad y_i\in \{+1,-1\} \\
\end{aligned}
\\
\text{Find } w \text{(normalized) and } b \\
\max_{w,b} \gamma \ \  s.t.\ \  y_i(w^Tx_i+b)\geq \gamma,\quad i=1,2,...,N \\
\text{To normalized } \gamma \text{ we divid both side with } \gamma: \\
\min_{w,b} \frac{1}{2}||w||^2 \ \ s.t.\ \ y_i(w^Tx_i+b)\geq 1,\quad i=1,2,...,N \\
$$

将含有不等式约束的凸二次规划问题，用拉格朗日乘子法得到无约束的拉格朗日目标函数

$$
L(w,b,\alpha)=\frac{1}{2}||w||^2-\sum_{i=1}^N\alpha_i(y_i(w^Tx_i+b)-1)
$$

上述式子存在一对对偶问题，分别是：
$$
\min_{w,b}\theta(w)=\min_{w,b}\max_{\alpha\geq0}L(w,b,\alpha)=p* \\
\max_{\alpha} A(\alpha)= \max_{\alpha}\min_{w,b}L(w,b,\alpha)=d*
$$
要求$d*=p*$需要满足优化问题是凸优化问题，满足 kkt 条件（kkt 条件是极值的必要条件）。本优化问题是一个1凸优化问题，所以条件一满足，要满足条件二，即要求：
$$
\left\{
\begin{aligned}
&\alpha_i \geq 0\\
&y_i(w_i^Tx_i+b)-1\geq 0 \\
&\alpha_i(y_i(w_i^Tx_i+b)- 1) = 0
\end{aligned}
\right.
$$
为了求解对偶问题的具体形式，令 $L(w,b,\alpha)$ 对 $w$ 和 b 的偏导为 0，可得：
$$
w=\sum_{i=1}^N\alpha_i y_i x_i \\
\sum_{i=1}^N \alpha_i y_i=0
$$
带入拉格朗日目标函数，消去 $w$ 和 $b$，得到：
$$
L(w,b,\alpha) = -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^T x_j) + \sum_{i=1}^N\alpha_i
$$
所以，对原拉格朗日的极小值问题：
$$
\min_{w,b}\max_{\alpha}L(w,b,\alpha) = -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^T x_j) + \sum_{i=1}^N\alpha_i
$$
可以变成对偶问题：
$$
\max_{\alpha}\min_{w,b}L(w,b,\alpha) = -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^T x_j) + \sum_{i=1}^N\alpha_i
$$
但是上式中的 $w$ 和 $b$ 均已被消去，只剩 $\alpha$，所以可得：
$$
\max_{\alpha} -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^T x_j) + \sum_{i=1}^N\alpha_i \\
s.t. \ \ \sum_{i=1}^N\alpha_i y_i=0 \\
\alpha_i \geq 0, \ \ i=1,2,...,N
$$
加上负号，取反之后变为极小值
$$
\min_{\alpha} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^T x_j) - \sum_{i=1}^N\alpha_i \\
s.t. \ \ \sum_{i=1}^N\alpha_i y_i=0 \\
\alpha_i \geq 0, \ \ i=1,2,...,N
$$
对于上述问题，我们有高效的优化算法，即序列最小化（SMO）算法。解出最优解 $\alpha^*$ 之后，自然可以求解出 $w$ 和 $b$，进而得到最初的目的，找到超平面。
$$
w^*=\sum_{i=1}^N\alpha_i*y_ix_i \\
b^*=\frac{1}{N}\sum_{i=1}^N (y_i-\sum_{j=1}^N\alpha_i^*y_j(x_i^Tx_j)) \text{ for } \alpha_i^* \not =0
\\
b^*=y_i-\sum_{j=1}^N\alpha_i^*y_j(x_i^Tx_j)\text{ for } \alpha_i^* \not =0
$$
分类决策函数为：
$$
f(x) = sign(w^{*T}x+b^*)
$$
上面对于 $b^*$ 的表达是等价的，相当于在可移动区间上做平移，以上都是说的是硬间隔。硬间隔适用于，原本样本全部可以正确划分，但是对于有噪音，或者数据本身就不能被一个超平面划分的情况，我们就考虑采取软间隔，引入松弛变量$1\geq\xi_i\geq0$
$$
y_i(w^Tx_i+b)\geq 1- \xi_i
$$
于是，加入软间隔最大化的 SVM 学习条件为：
$$
\min_{w}\frac{1}{2}||w||_2^2+C\sum_{i=1}^N\xi_i \\
s.t. \ \ y_i(w^Tx_i+b)\geq 1 - \xi_i \ \ (i=1,2,...,N) \\
\xi_i\geq 0 \ \ (i=1,2,...,m)
$$
这里 $C$ 做为惩罚参数，可以理解为正则化参数，来对误分类的惩罚。同理，我们可以写出软间隔条件下，拉格朗日函数的无约束优化问题：

$$
L(w,b,\xi,\alpha,\mu)=\frac{1}{2}||w||_2^2+C\sum_{i=1}^N\xi_i-\sum_{i=1}^N\alpha_i(y_i(w^Tx_i+b)-1+\xi_i)-\sum_{i=1}^N\mu_i\xi_i
$$
其中 $\mu_i\geq 0$ 和 $\alpha_i \geq 0 $ 为拉格朗日系数，所以我们的对偶问题为：
$$
\min_{w,b,\xi}\max_{\alpha,\mu} L(w,b,\alpha,\xi,\mu) \\
\max_{\alpha,\mu}\min_{w,b,\xi} L(w,b,\alpha,\xi,\mu)
$$
同样的，对于原来的 $L$，对 $w,b,\xi$ 求偏导令其为0，可得到：
$$
w=\sum_{i=1}^N\alpha_iy_ix_i \\
0=\sum_{i=1}^N\alpha_iy_i \\
C-\alpha_i-\mu_i=0
$$
利用上述三个式子，消除 $w$ 和 $b$
$$
L(w,b,\xi,\alpha,\mu)=\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^Tx_j)
$$
同样的，由对偶问题，我们可以得到现在需要优化目标1的数学形式：
$$
\max_{\alpha} \ \sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^Tx_j) \\
s.t. \ \ \sum_{i=1}^N\alpha_iy_i=0 \\
C-\alpha_i-\mu_i=0\\
\alpha_i\geq0\ \ (i=1,2,...,N) \\
\alpha_i\geq0\ \ (i=1,2,...,N) \\ 
$$
消去 $\mu_i$，只留下 $\alpha_i$
$$
\min_{\alpha} \ \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^Tx_j) -\sum_{i=1}^N\alpha_i \\
s.t. \ \ \sum_{i=1}^N\alpha_iy_i=0 \\
0\leq\alpha_i\leq C
$$
最后解出来 $\alpha^*$，
$$
\begin{aligned}
&\alpha^*_i = 0, \text{被正确分类，而且原理支持向量点} \\
&0 \leq \alpha^*_i \le C, \xi_i=0 \text{， 在支持向量上} \\
&\alpha^*_i = C, 0\le \xi_i\leq1 \text{， 正确分类，但是在超平面和支持向量之间} \\
&\alpha^*_i = C, \xi=1 \text{， 在超平面上，无法被正确分类} \\
&\alpha^*_i = C, \xi > 1 \text{，在超平面另一侧，无法被正常分类} \\
\end{aligned}
$$
最终的决策函数和硬间隔一样，也为
$$
f(x)=sign(w^{*T}x+b^*)
$$

核函数的引入，对于线性不可分的低维度特征数据，我们可以将其映射到高维，就能够线性可分，这个映射我们称为 $\phi()$，然后由某一个定理的保证下，在这个映射满足某些条件之下。我们存在一个核函数 $K$ 使得：
$$
K(x,y)=\phi(x)^T\phi(y)
$$
常用的核函数有高斯核函数，径向基核函数（RBF）其中后者是最常用的，表达式为
$$
K(x,y)=exp(-\gamma||x-y||^2)
$$
多项式核函数
$$
K(x,y)=(\gamma x^Ty+r)^d
$$
sigmoid 核函数
$$
K(x,y)=tanh(\gamma x^Ty+r)
$$

除了分类任务，SVM 同样也可以做为回归模型。假设我们的模型判别函数为 $\hat{y_i}=w^T\phi(x_i)+b$，那么我们采用的回归模型的损失函数度量为：
$$
err(x_i,y_i)=
\left\{
\begin{aligned}

0, |y_i-w^T\phi(x_i)-b|\leq \epsilon \\
|y_i-w^T\phi(x_i)-b|-\epsilon, |y_i-w^T\phi(x_i)-b|\ge \epsilon \\
\end{aligned}
\right. 
$$
所以我们的目标函数定义为：
$$
\min_w \frac{1}{2}||w||_2^2 \\ s.t.\ \ |y_i-w^T\phi(x_i)-b|\leq\epsilon\ \ (i=1,2,...,N)
$$
和分类模型类似，回归模型也同样可以对每个样本加入松弛变量，但是由于我们这里回归模型的损失，加上了绝对值，所以我们需要正负两个松弛变量

$$
\min\_w\frac{1}{2}||w||_2^2+C\sum_{i=1}^N(\xi_i^++\xi_i^-) \\
s.t.\ \ -\epsilon-\xi_i^-\leq y_i - w^T\phi(x_i)-b\leq \epsilon +\xi_i^+ \\
\xi_i^+\geq0,\ \xi_i^-\geq0, \ \ (i=1,2,...,N)
$$
同理我们可以得到一对拉格朗日对偶问题的表达式：
$$
\min_{wb,\xi^+,\xi_-}\max_{\mu^+\geq0,\mu^-\geq0,\alpha^+\geq0, \alpha^-\geq0,}L(w,b,\alpha^+,\alpha^-,\xi^+,\xi^-,\mu^+,\mu^-) \\
\max_{\mu^+\geq0,\mu^-\geq0,\alpha^+\geq0, \alpha^-\geq0,}\min_{wb,\xi^+,\xi_-}L(w,b,\alpha^+,\alpha^-,\xi^+,\xi^-,\mu^+,\mu^-)
$$
通过对拉格朗日函数对$w,b,\xi^+,\xi^-$求偏导数令为 0，可以得到：
$$
w = \sum_{i=1}^N(\alpha_i^+ - \alpha_i^-)\phi(x_i) \\
0 = \sum_{i=1}^N(\alpha_i^+ - \alpha_i^-) \\
C-\alpha_i^+ - \mu^+_i = 0 \\
C - \alpha_i^- - \mu^-_i = 0
$$
带回拉格朗日函数中，消去$w,b,\xi^+,\xi^-$，可以得到最终的对偶形式为：
$$
\max_{\alpha^+,\alpha^-}\sum_{i=1}^N(\epsilon - y_i)\alpha_i^++(\epsilon+y_i)\alpha_i^- - \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N(\alpha_i^+-\alpha_i^-)(\alpha_j^+-\alpha_j^-)K(x_i, x_j) \\
s.t. \ \ \sum_{i=1}^N(\alpha_i^+ - \alpha_i^-) = 0 \\
0 < \alpha_i^+ < C, \ \ (i=1,2,...,N) \\
0 < \alpha_i^- < C, \ \ (i=1,2,...,N)
$$
对目标函数取负号，求最小值可以得到和 VSM 分类模型类似的求极小值的目标函数为：
$$
\min_{\alpha^+, \alpha^-}\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N(\alpha_i^+-\alpha_i^-)(\alpha_j^+-\alpha_j^-)K(x_i, x_j) - \sum_{i=1}^N(\epsilon - y_i)\alpha_i^++(\epsilon+y_i)\alpha_i^- \\
s.t. \ \ \sum_{i=1}^N(\alpha_i^+ - \alpha_i^-) = 0 \\
0 < \alpha_i^+ < C, \ \ (i=1,2,...,N) \\
0 < \alpha_i^- < C, \ \ (i=1,2,...,N)
$$
对于这个目标函数，我们依然可以用 SMO 算法来求出对应的 $\alpha^+$ 和 $\alpha^-$ 进而求出最终的回归系数 $w, b$


参考：
https://zhuanlan.zhihu.com/p/29862011