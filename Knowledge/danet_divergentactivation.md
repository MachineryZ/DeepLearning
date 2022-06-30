# DA Net

DANet: Divergent Activation For Weakly Supervised Object Localization

https://openaccess.thecvf.com/content_ICCV_2019/papers/Xue_DANet_Divergent_Activation_for_Weakly_Supervised_Object_Localization_ICCV_2019_paper.pdf

本文 task 主要是 weakly supervised object localization。为了提高像素级识别表示的辨别能力：
1. 利用多尺度上下文融合，结合不同的 dilated conv 和 池化操作
2. 使用分解结构，增大卷积核尺寸或在网络顶部引入有效的编码层，来捕获更丰富的全局信息
3. encoder-decoder 结构来融合中级和高级语义特征
4. 使用 rnn 捕捉长城依赖关系，从而提高分割精度




