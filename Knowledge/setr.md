# Setr

Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformer
https://arxiv.org/pdf/2012.15840.pdf


传统 cnn 网络在 segmentation 领域已经取得了很多成功，于是 transformer 也进来插了一脚。那么，对于 transformer 的输入就需要一个 sequence 的输入，所以第一步就是把图像输入变成 sequence 输入。


SETR tensor 转换过程：
1. 输入为 $x \in R^{H\times W\times 3}$
2. downsampple 到一个更小的维度 $x_f\in R^{H/16 \times W/16\times C}$，所以序列长度为 $HW/256$
3. 然后就是正常的过 transformer 的 encoder 即可。



