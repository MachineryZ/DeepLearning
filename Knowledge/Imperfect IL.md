# Imperfect IL

Learning from Imperfect demonstrations from Agents with Varying Dynamics

https://arxiv.org/pdf/2103.05910.pdf

本篇 paper 是一次面试中，候选人的 imperfect imitation learning 系列 project 的一些

Imitation Learning: Imitation Learning 的出现本质是为了解决 reinforcement learning 里一些问题：
1. reinforcement learning 的样本需要与 simulator 进行交互，那么对于非常精细的任务就需要更高精度的、更贴近现实情况的 simulator
2. reinforcement learning 的 policy net 的训练是非常困难的，因为需要大量样本，也需要非常好的设计去让 policy 学到一个 optimal 的策略，所以 reinforcement learning 的 training 有两大特点：需要大量数据，训练容易不收敛
3. reinforcement learning 的训练需要一个明确的 reward function，但是对于非常抽象的任务来说，其实一般没办法做的十分量化的定义出来一个“好坏”

Imitation Learning 的出现很好的解决了以上的问题：
1. 对于 simulator 的问题，很多 imitation learning 的 demonstration 其实都是人为操作的机械臂，或者说，叫 agent 来得到的一个 trajectory，这样实际上是和真实环境直接交互
2. 同理，如果我们的 demonstration 都是较好的数据点，那么在 training 的时候就不会有太多噪声，这样也不会导致网络收敛性的波动问题，而且也就只需要稍微少量的数据就可以进行成功的一次 training
3. imitation learning 的目标其实并不是优化 reward function，而是让当前的 model 去学习一个更加接近 demonstration 的 policy 的过程，本质是让两个概率分布更加靠近，所以也刚好绕开了这个 reward function 的设置问题

但是，imitation learning 会出现一个问题，就是，人为、手动操作得到的 demonstration 并不一定是最优的，它有可能是次优的，我们这里称为 imperfect demonstration，那么如何从 imperfect demonstration 得到一个比较好的策略，是本篇 paper 在考虑的事情

本篇 paper 的想法也比较简单，我们有一个 ranking 的算法，但是不是对两个 demonstration 的好坏打分，否则相当于我们直接有了一个能够评价 demonstration 好坏的算法，那我们就可以用这个算法去做监督训练了。而是对于某一个 demonstration 的某一个环节进行打分。

有了这个算法之后，我们可以在 trajectory 上进行 q-learning 训练，然后得到一个 policy net，就是最终结果了。

