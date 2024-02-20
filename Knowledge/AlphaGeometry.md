# AlphaGeometry
https://www.nature.com/articles/s41586-023-06747-5

大致逻辑和思路
- alphageometry 模型的目标就是解决用计算机算法解决复杂几何问题（IMO 的几何问题）
- 解决几何问题的难点在于，数据量稀疏，人类解决几何问题是用较强的推理能力（和各类定理）而不是所谓用所谓的大量数据训练，非常不适合 ai 去解决。说白了就是，逻辑问题是 learning-based 而不是 data-based 的
-  
- 一些可能存在的问题（第一时间我想到如何去做这个 task，感觉会遇到的问题）
    - 几何图形如何存储成为计算机中的 tensor 类的数据类型，如 circle，triangle，是进行特定的编码么？
    - 如何处理不同图形之间的关系，比如两个线段相交、垂直、还是平行？
    - 定理如何表述，用一些特殊的矩阵来表示？？
    - 训练数据如何获得，比如某个几何问题可以有很多种证明方法，是用多个过程都送入训练么？如此稀少的几何题证明数据，如何增加、生成数据？
    - 如何保证证明的正确性？防止伪证，如何检查？以此来保证证明的完整性
    - 如何评价一个证明的好坏（评价指标），是证明 steps 么？

synthetic theorems and proofs generation
- 合成前提和证明生成
    - 合成前提，论文的意思好像是随机生成了百万级别的假设？（并没有用已有的几何问题，而是任意生成一个图）
    - 生成一个训练的三元要素 （premises，conclusion，proof）
    - 用 symbolic deduction engine 去做推导（这些都是前人有过工作的）
    - 加入了解析几何的方法，去拓展生成的定理和证明

generating proofs beyond symbolic deduction
- auxiliary construction 辅助结构（辅助线、辅助圆等）
    - 这个好处是可以在当前条件下生成无数的可以推导的情况，让模型去搜索
    - 不需要 human demonstration 这样也能学会做辅助结构
- training languag model
    - 模型选择依然是 transformer 类模型，因为对于text sequence 类型的数据，transformer 有天然的优势
    - 
geometry theorem prover baselines
- 几何问题 solver
    - 第一种是解析几何强行解，解析几何 solver 的方法通常没有逻辑推导，需要的是大量的内存和计算复杂度
    - 第二种是 search/axiomatic methods，这种方法是用若干个定理去推导得到结果

synthetic theorems rediscover
- 一些随机生成的 synthetic theorems 可以发现很多 theorem precises（这一般是得益于做的辅助结构）
- 虽然这些 synthetic theorem 的证明一般会比 imo 最难的题目还要长 30%
- 但是 synthetic theorem 并不在目前 discovered theorem set 中，为后续的证明提供了很好的基础

Language model pretraining and fine-tuning
- pretrain on 100 million synthetic theorem data
- fine-tuning model 在需要做辅助线的训练集子集里，大概 9%，也就是 9 million

Proving results on IMO-AG-30
- 就一些结果对比吧，和不同方法的证明准确率对比

Geometry Representation
- 原话：To sidestep this barrier, we instead adopted a more specialized language used in GEX10, JGEX17, MMP/ Geometer13 and GeoLogic19, a line of work that aims to provide a logi- cal and graphical environment for synthetic geometry theorems with human-like non-degeneracy and topological assumptions
- gex 这个是一个 paper 中专门用数学符号来构建集合学语言的一个系统，说白了就是采用了一些特定的已经确定好的自然语言系统来表示图形

Sampling consistent theorem premises

Algebraic Reasoning：
- 把一些角度的表示表示为两个线段和 x-axis 的角度差，然后就可以有解析解的形式
