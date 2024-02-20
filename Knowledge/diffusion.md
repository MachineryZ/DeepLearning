# Diffusion

Denoising Diffusion Probabilistic Models

https://arxiv.org/abs/2006.11239

这篇 markdown 主要介绍和总结 diffusion model 的应用。

Diffusion Model（扩散模型）是一种生成模型，它基于随机过程来生成数据。这种模型的核心思想是将数据生成过程看作是一个逐步增加噪声的扩散过程，然后通过反向过程（即去噪过程）来生成数据。

**扩散过程**

也即，往图片加上噪声的过程。给定图片 $x_0$，前向 T 次雷击对其添加高斯噪声，得到 $x_1,x_2,...,x_T$，前向过程是马尔可夫过程
$$
q(x_t|x_{t-1}) = \mathcal N(x_t;\sqrt{q-\beta_t}x_{t-1},\beta_t\mathbf I)\\
q(x_{1:T}|x_0)=\Pi_{t=1}^Tq(x_t|x_{t-1}) = \Pi^T_{t-1}\mathcal N(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$
**逆扩散过程**

从标准高斯分布 $\mathcal N(0,\mathbf I)$ 还原出样本数据的分布 $x_0$ ，从纯高斯噪声 $p(x_T) := \mathcal N(x_T;0,\mathbf I)$ 开始，模型将学习联合概率分布 $p_\theta(x_{T:0})$  
$$
p_\theta(x_T:0) := p(x_T)\Pi^T_{t=1}p_\theta(x_{t-1}|x_t)\\
=p(x_t)\Pi_{t=1}^T\mathcal N(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$
（网络需要学习的就是均值和方差，以此来去除噪声，重构图片）

---

损失函数
$$
||u-\epsilon_\theta(\bar \alpha_tx_0 + \bar\beta_tu,t)||_2^2
$$


超参数设置 $\alpha_t = \sqrt{1 - 0.02t/T}$



---

代码实现

~~~python
num_steps = 1000
beta = torch.tensor(np.linspace(1e-5, 0.2e-2, num_steps))

alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alpha_bar_log = torch.log(1 - alphas_prod)
one_minus_alpha_bar_sqrt = torch.sqrt(1 - alphas_prod)

def q_x(x_0, t, noise = None):
  if not noise:
    noise = torch.randn_like(x_0)
  alphas_t = extract(alphas_bar_sqrt, t, x_0)
  alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
  return (alphas_t * x_0 + alphas_1_m_t * noise)

~~~



















































