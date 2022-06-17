# PointNet++

之前我们提到过的 PointNet 有以下几个问题：
1. point-wise MLP，仅仅是对每个点表征，对局部结构信息整合能力太弱
2. global feature 直接由 max pooling 获得，无论是对分类还是分割任务，都会造成巨大的信息损失
3. 分割任务的全局特征 global feature 是直接复制与 local feature 进行 concat，生成 discriminative feature 能力有限。

PointNet++ 对上述几个问题的解决办法是：
1. sampling 和 grouping 整合局部邻域
2. 利用 hierarchical feature learning framework，通过多个 set abstraction 逐级降采样，获得不同规模、不同层次的 local-global feature
3. 对分割任务设计了 encoder-decoder 结构，先降采样再上采样，使用 skip connection 将对应层的 local-global feature 拼接


<div align=center><img src="../Files/pointnet++1.jpg" width=80%></div>
<div align=center><img src="../Files/pointnet++2.jpg" width=80%></div>

首先讨论最远点采样算法（farthest point sampling，FPS），先选定部分点作为局部区域的中心，通过设定需要采样的点云数量来控制模型的计算量，再通过后续网络学习点的局部特征。