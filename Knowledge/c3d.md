# C3D

ICCV 15

在 two stream 里，optical flow 是非常有用的。但是 optical flow 的计算是非常麻烦的，费时间、费内存、而且 infer 的速度也非常慢。所以，就有了 3d cnn 的出现。3d cnn 在时间域和空间域上是同时建模，那么就没有必要再用光流去特别的代替 temporal 的信息了，大大加速了计算所需要的资源。但是之前的 3d cnn 的效果一直不行，直到本篇的 c3d 的出现。