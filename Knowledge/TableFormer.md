# TableFormer

TableFormer: Table Structure Understanding with Transformers

https://arxiv.org/pdf/2203.01017.pdf

如果是我看到这篇 paper 的话，我会问这么几个问题：
1. 表格数据和 图片、nlp 的 数据本质上有什么区别
2. 表格数据在用规整化 tensor 表达是怎么表达的
3. 在 model 方面，table former 结构是什么
4. table 数据，有哪些数据集

数据集有 PubTabNet，论文原文 https://arxiv.org/pdf/1911.10683.pdf，tableformer 也同样提出了一个 table structure dataset 叫 SynthTabNet

表格以简洁紧凑的方式组织有价值的内容。这些内容对于搜索引擎、知识图等系统非常有价值，因为它们增强了它们的预测能力。不幸的是，表格有各种各样的形状和大小。此外，它们可能具有复杂的列/行标题配置、多行行、不同种类的分隔线、缺少条目等。因此，从图像中正确识别表格结构是一项不容易的任务