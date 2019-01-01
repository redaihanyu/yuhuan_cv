排版问题，另请参考：[https://shimo.im/docs/Bcpj57sUS8gsdbBa/](https://shimo.im/docs/Bcpj57sUS8gsdbBa/) 《编码学习Faster R-CNN》，可复制链接后用石墨文档 App 打开

Copy from : [https://zhuanlan.zhihu.com/p/32404424](https://zhuanlan.zhihu.com/p/32404424)
Faster R-CNN的极简实现： [github: simple-faster-rcnn-pytorch](https://link.zhihu.com/?target=https%3A//github.com/chenyuntc/simple-faster-rcnn-pytorch)
本文插图地址（含五幅高清矢量图）：[draw.io](https://link.zhihu.com/?target=https%3A//www.draw.io/%3Flightbox%3D1%26highlight%3D0000ff%26edit%3D_blank%26layers%3D1%26nav%3D1%26title%3Dfaster-rcnn%25E7%259A%2584%25E5%2589%25AF%25E6%259C%25AC%25E7%259A%2584%25E5%2589%25AF%25E6%259C%25AC.xml%23Uhttps%253A%252F%252Fraw.githubusercontent.com%252Fchenyuntc%252Fcloud%252Fmaster%252Ffaster-rcnn%2525E7%25259A%252584%2525E5%252589%2525AF%2525E6%25259C%2525AC%2525E7%25259A%252584%2525E5%252589%2525AF%2525E6%25259C%2525AC.xml)
# 1 概述
在目标检测领域, Faster R-CNN表现出了极强的生命力, 虽然是2015年的[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1506.01497), 但它至今仍是许多目标检测算法的基础，这在日新月异的深度学习领域十分难得。Faster R-CNN还被应用到更多的领域中, 比如人体关键点检测、目标追踪、 实例分割还有图像描述等。

现在很多优秀的Faster R-CNN博客大都是针对论文讲解，本文将尝试从编程角度讲解Faster R-CNN的实现。由于Faster R-CNN流程复杂，符号较多，容易混淆，本文以VGG16为例，所有插图、数值皆是基于VGG16+VOC2007 。
## 1.1 目标
从编程实现角度角度来讲, 以Faster R-CNN为代表的Object Detection任务，可以描述成:
给定一张图片, 找出图中的有哪些对象,以及这些对象的位置和置信概率。
![图片](https://pic2.zhimg.com/80/v2-79ee2d46ba80b773089056b69b55b991_hd.jpg)目标检测任务
## 1.2 整体架构
Faster R-CNN的整体流程如下图所示。
![图片](https://pic2.zhimg.com/80/v2-4e372e4536ef6d3d28ebd8803a9b13e2_hd.jpg)Faster R-CNN整体架构
从编程角度来说， Faster R-CNN主要分为四部分（图中四个绿色框）：
* Dataset：数据，提供符合要求的数据格式（目前常用数据集是VOC和COCO）
* Extractor： 利用CNN提取图片特征features（原始论文用的是ZF和VGG16，后来人们又用ResNet101）
* RPN(*Region Proposal Network): *负责提供候选区域rois（每张图给出大概2000个候选框）
* RoIHead： 负责对rois分类和微调。对RPN找出的rois，判断它是否包含目标，并修正框的位置和座标

Faster R-CNN整体的流程可以分为三步：
* 提特征： 图片（img）经过预训练的网络（Extractor），提取到了图片的特征（feature）
* Region Proposal： 利用提取的特征（feature），经过RPN网络，找出一定数量的rois（region of interests）。
* 分类与回归：将rois和图像特征features，输入到RoIHead，对这些rois进行分类，判断都属于什么类别，同时对这些rois的位置进行微调。
# 2 实现
## 2.1 数据
对与每张图片，需要进行如下数据处理：
* 图片进行缩放，使得长边小于等于1000，短边小于等于600（至少有一个等于）。
* 对相应的bounding boxes 也进行同等尺度的缩放。
* 对于Caffe 的VGG16 预训练模型，需要图片位于0-255，BGR格式，并减去一个均值，使得图片像素的均值为0。

最后返回四个值供模型训练：
* images ： 3×H×W ，BGR三通道，宽W，高H
* bboxes： 4×K , K个bounding boxes，每个bounding box的左上角和右下角的座标，形如（Y_min,X_min, Y_max,X_max）,第Y行，第X列。
* labels：K， 对应K个bounding boxes的label（对于VOC取值范围为[0-19]）
* scale: 缩放的倍数, 原图H' ×W'被resize到了HxW（scale=H/H' ）

需要注意的是，目前大多数Faster R-CNN实现都只支持batch-size=1的训练（[这个](https://zhuanlan.zhihu.com/github.com/jwyang/faster-rcnn.pytorch) 和[这个](https://link.zhihu.com/?target=https%3A//github.com/precedenceguo/mx-rcnn)实现支持batch_size>1）。
## 2.2 Extractor
Extractor使用的是预训练好的模型提取图片的特征。论文中主要使用的是Caffe的预训练模型VGG16。修改如下图所示：为了节省显存，前四层卷积层的学习率设为0。Conv5_3的输出作为图片特征（feature）。conv5_3相比于输入，下采样了16倍，也就是说输入的图片尺寸为3×H×W，那么feature的尺寸就是C×(H/16)×(W/16)。VGG最后的三层全连接层的前两层，一般用来初始化RoIHead的部分参数，这个我们稍后再讲。总之，一张图片，经过extractor之后，会得到一个C×(H/16)×(W/16)的feature map。
![图片](https://pic2.zhimg.com/80/v2-28887eb4f69439e1384165da0ca20b6f_hd.jpg)
Extractor: VGG16
## 2.3 RPN
Faster R-CNN最突出的贡献就在于提出了Region Proposal Network（RPN）代替了Selective Search，从而将候选区域提取的时间开销几乎降为0（2s -> 0.01s）。
### 2.3.1 Anchor
在RPN中，作者提出了anchor。Anchor是大小和尺寸固定的候选框。论文中用到的anchor有三种尺寸和三种比例，如下图所示，三种尺寸分别是小（蓝128）中（红256）大（绿512），三个比例分别是1:1，1:2，2:1。3×3的组合总共有9种anchor。
![图片](https://pic1.zhimg.com/80/v2-7abead97efcc46a3ee5b030a2151643f_hd.jpg)
Anchor

然后用这9种anchor在特征图（feature）左右上下移动，每一个特征图上的点都有9个anchor，最终生成了 (H/16)× (W/16)×9个anchor. 对于一个512×62×37的feature map，有 62×37×9约20, 000个anchor。 也就是对一张图片，有20000个左右的anchor。这种做法很像是暴力穷举，20000多个anchor，哪怕是蒙也能够把绝大多数的ground truth bounding boxes蒙中。
### 2.3.2 训练RPN
RPN的总体架构如下图所示：
![图片](https://pic2.zhimg.com/80/v2-e7eeb94a86ece2dadfa9db2277f7d016_hd.jpg)RPN架构
anchor的数量和feature map相关，不同的feature map对应的anchor数量也不一样。RPN在Extractor输出的feature maps的基础之上，先增加了一个卷积（用来语义空间转换？），然后利用两个1x1的卷积分别进行二分类（是否为正样本）和位置回归。

进行分类的卷积核通道数为9×2（9个anchor，每个anchor二分类，使用交叉熵损失），进行回归的卷积核通道数为9×4（9个anchor，每个anchor有4个位置参数）。RPN是一个全卷积网络（fcnn, fully convolutional network），这样对输入图片的尺寸就没有要求了。

接下来RPN做的事情就是利用（AnchorTargetCreator）将20000多个候选的anchor选出mini-batch=256个anchor进行分类和回归位置。选择过程如下：
* 对于每一个ground truth bounding box (gt_bbox)，选择和它重叠度（IoU）最高的一个anchor作为正样本，这样能保证每一个bbox至少有一个正样本。
* 对于剩下的anchor，从中选择和任意一个gt_bbox重叠度超过0.7的anchor，作为正样本，正样本的数目不超过128个。
* 随机选择和gt_bbox重叠度小于0.3的anchor作为负样本。负样本和正样本的总数为256。

对于每个anchor, gt_label 要么为1（前景），要么为0（背景），而gt_loc则是由4个位置参数(tx,ty,tw,th)组成，这样比直接回归座标更好。
![图片](https://images-cdn.shimo.im/YWZW004hRGgJE0oi/image.png!thumbnail)
注：ti 和 ti* 分别为网络的预测值和回归的目标。

![图片](https://www.zhihu.com/equation?tex=t_x+%3D+%28x+%E2%88%92+x_a%29%2Fw_a%3B+t_y+%3D+%28y+%E2%88%92+y_a%29%2Fh_a%3B%5C%5C+t_w+%3D+log%28w%2Fw_a%29%3B+t_h+%3D+log%28h%2Fh_a%29%3B%5C%5C+t_x%5E%2A+%3D+%28x%5E%2A+%E2%88%92+x_a%29%2Fw_a%3B+t_y%5E%2A+%3D+%28y%5E%2A+%E2%88%92+y_a%29%2Fh_a%3B%5C%5C+t_w%5E%2A+%3D+log%28w%5E%2A%2Fw_a%29%3B+t_h%5E%2A+%3D+log%28h%5E%2A%2Fh_a%29%3B%5C%5C)
注：其中 x, x_a, x*分别对应predicted box, anchor box和ground-truth box。

目标 t* 是通过 ground-truth box（目标真实box）和 anchor box 计算得出的，代表的是ground-truth box 与 anchor box 之间的转化关系。用这个来训练 rpn，那么 rpn 最终学会输出一个良好的转化关系 t ，而 t 是 predicted box 与 anchor box 之间的转化关系。通过这个 t和 anchor box ，可以计算出 predicted box 的真实坐标。

计算分类损失用的是交叉熵损失，而计算回归损失用的是 Smooth_l1_loss。在计算回归损失的时候，只计算正样本（前景）的损失，不计算负样本的位置损失。
![图片](https://images-cdn.shimo.im/2pdddro6osEf9Ltl/image.png!thumbnail)
其中，smooth_L1_loss 表示为：
![图片](https://images-cdn.shimo.im/38pFaGSeVugnSSKO/image.png!thumbnail)
### 2.3.3 RPN生成RoIs
RPN在自身训练的同时，还会提供RoIs（region of interests）给Fast RCNN（RoIHead）作为训练样本。

RPN生成RoIs的过程(ProposalCreator)如下：
* 对于每张图片，利用它的feature map， 计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景的概率，以及对应的位置参数。
* 选取概率较大的12000个anchor，利用回归的位置参数，修正这12000个anchor的位置，得到RoIs利用非极大值（(Non-maximum suppression, NMS）抑制，选出概率最大的2000个RoIs

注意：在 inference 的时候，为了提高处理速度，12000和2000分别变为6000和300.
注意：这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现。
RPN的输出：RoIs（形如2000×4或者300×4的tensor）
## 2.4 RoIHead/Fast R-CNN
RPN只是给出了2000个候选框，RoI Head在给出的2000候选框之上继续进行分类和位置参数的回归。
### 2.4.1 网络结构
![图片](https://pic1.zhimg.com/80/v2-5b0d1ca6e990fcdecd41280b69cd8622_hd.jpg)
RoIHead网络结构

由于RoIs给出的2000个候选框，分别对应feature map不同大小的区域。首先利用ProposalTargetCreator 挑选出128个sample_rois, 然后使用了RoIPooling 将这些不同尺寸的区域全部pooling到同一个尺度（7×7）上。

下图就是一个例子，对于feature map上两个不同尺度的RoI，经过RoIPooling之后，最后得到了3×3的feature map.
![图片](https://pic1.zhimg.com/80/v2-d9eb14da175f7ae2ed6b6d77f8993207_hd.jpg)RoIPooling
RoI Pooling 是一种特殊的Pooling操作，给定一张图片的Feature map (512×H/16×W/16) ，和128个候选区域的座标（128×4），RoI Pooling将这些区域统一下采样到 （512×7×7），就得到了128×512×7×7的向量。可以看成是一个batch-size=128，通道数为512，7×7的feature map。

为什么要pooling成7×7的尺度？是为了能够共享权重。在之前讲过，除了用到VGG前几层的卷积之外，最后的全连接层也可以继续利用。当所有的RoIs都被pooling成（512×7×7）的feature map后，将它reshape 成一个一维的向量，就可以利用VGG16预训练的权重，初始化前两层全连接。最后再接两个全连接层，分别是：
* FC 21 用来分类，预测RoIs属于哪个类别（20个类+背景）
* FC 84 用来回归位置（21个类，每个类都有4个位置参数）
### 2.4.2 训练
前面讲过，RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，而是利用ProposalTargetCreator 选择128个RoIs用以训练。选择的规则如下：
* RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）
* 选择 RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本

为了便于训练，对选择出的128个RoIs，还对他们的gt_roi_loc 进行标准化处理（减去均值除以标准差）

对于分类问题,直接利用交叉熵损失，而对于位置的回归损失,一样采用Smooth_L1_Loss, 只不过只对正样本计算损失。而且是只对正样本中的这个类别4个参数计算损失。举例来说:
* 一个RoI在经过FC 84后会输出一个84维的loc 向量. 如果这个RoI是负样本,则这84维向量不参与计算 L1_Loss
* 如果这个RoI是正样本,属于label K,那么它的第 K×4, K×4+1 ，K×4+2， K×4+3 这4个数参与计算损失，其余的不参与计算损失。
### 2.4.3 生成预测结果
测试的时候对所有的RoIs（大概300个左右) 计算概率，并利用位置参数调整预测候选框的位置。然后再用一遍极大值抑制（之前在RPN的ProposalCreator用过）。

注意：
* 在RPN的时候，已经对anchor做了一遍NMS，在RCNN测试的时候，还要再做一遍
* 在RPN的时候，已经对anchor的位置做了回归调整，在RCNN阶段还要对RoI再做一遍
* 在RPN阶段分类是二分类，而Fast RCNN阶段是21分类
## 2.5 模型架构图
最后整体的模型架构图如下：
![图片](https://pic4.zhimg.com/80/v2-7c388ef5376e1057785e2f93b79df0f6_hd.jpg)整体网络结构

需要注意的是： 蓝色箭头的线代表着计算图，梯度反向传播会经过。而红色部分的线不需要进行反向传播（论文了中提到了ProposalCreator生成RoIs的过程也能进行反向传播，但需要专门的[算法](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1512.04412)）。
# 3 概念对比
在Faster RCNN中有几个概念，容易混淆，或者具有较强的相似性。在此我列出来并做对比，希望对你理解有帮助。
## 3.1 bbox anchor RoI loc
**BBox**：全称是bounding box，边界框。其中Ground Truth Bounding Box是每一张图中人工标注的框的位置。一张图中有几个目标，就有几个框(一般小于10个框)。Faster R-CNN的预测结果也可以叫bounding box，不过一般叫 Predict Bounding Box.
**Anchor**：锚？是人为选定的具有一定尺度、比例的框。一个feature map的锚的数目有上万个（比如 20000）。
**RoI**：region of interest，候选框。Faster R-CNN之前传统的做法是利用selective search从一张图上大概2000个候选框框。现在利用RPN可以从上万的anchor中找出一定数目更有可能的候选框。在训练RCNN的时候，这个数目是2000，在测试推理阶段，这个数目是300（为了速度）*我个人实验发现RPN生成更多的RoI能得到更高的mAP。*

RoI不是单纯的从anchor中选取一些出来作为候选框，它还会利用回归位置参数，微调anchor的形状和位置。
可以这么理解：在RPN阶段，先穷举生成千上万个anchor，然后利用Ground Truth Bounding Boxes，训练这些anchor，而后从anchor中找出一定数目的候选区域（RoIs）。RoIs在下一阶段用来训练RoIHead，最后生成Predict Bounding Boxes。

**loc**： bbox，anchor和RoI，本质上都是一个框，可以用四个数（y_min, x_min, y_max, x_max）表示框的位置，即左上角的座标和右下角的座标。这里之所以先写y，再写x是为了数组索引方便，但也需要千万注意不要弄混了。 我在实现的时候，没注意，导致输入到RoIPooling的座标不对，浪费了好长时间。除了用这四个数表示一个座标之外，还可以用（y，x，h，w）表示，即框的中心座标和长宽。在训练中进行位置回归的时候，用的是后一种的表示。
## 3.2 四类损失
虽然原始论文中用的4-Step Alternating Training 即四步交替迭代训练。然而现在github上开源的实现大多是采用近似联合训练（Approximate joint training），端到端，一步到位，速度更快。

在训练Faster RCNN的时候有四个损失：
* RPN 分类损失：anchor是否为前景（二分类）
* RPN位置回归损失：anchor位置微调RoI 
* 分类损失：RoI所属类别（21分类，多了一个类作为背景）
* RoI位置回归损失：继续对RoI位置微调

四个损失相加作为最后的损失，反向传播，更新参数。
## 3.3 三个creator
在一开始阅读源码的时候，我常常把Faster RCNN中用到的三个Creator弄混。
* AnchorTargetCreator ： 负责在训练RPN的时候，从上万个anchor中选择一些(比如256)进行训练，以使得正负样本比例大概是1:1. 同时给出训练的位置参数目标。 即返回gt_rpn_loc和gt_rpn_label。
* ProposalTargetCreator： 负责在训练RoIHead/Fast R-CNN的时候，从RoIs选择一部分(比如128个)用以训练。同时给定训练目标, 返回（sample_RoI, gt_RoI_loc, gt_RoI_label）
* ProposalCreator： 在RPN中，从上万个anchor中，选择一定数目（2000或者300），调整大小和位置，生成RoIs，用以Fast R-CNN训练或者测试。

其中AnchorTargetCreator和ProposalTargetCreator是为了生成训练的目标，只在训练阶段用到，ProposalCreator是RPN为Fast R-CNN生成RoIs，在训练和测试阶段都会用到。三个共同点在于他们都不需要考虑反向传播（因此不同框架间可以共享numpy实现）
## 3.4 感受野与scale
从直观上讲，感受野（*receptive field*）就是视觉感受区域的大小。在卷积神经网络中，感受野的定义是卷积神经网络每一层输出的特征图（feature map）上的像素点在原始图像上映射的区域大小。我的理解是，feature map上的某一点f对应输入图片中的一个区域，这个区域中的点发生变化，f可能随之变化。而这个区域外的其它点不论如何改变，f的值都不会受之影响。VGG16的conv5_3的感受野为228，即feature map上每一个点，都包含了原图一个228×228区域的信息。

Scale：输入图片的尺寸比上feature map的尺寸。比如输入图片是3×224×224，feature map 是 512×14×14，那么scale就是 14/224=1/16。可以认为feature map中一个点对应输入图片的16个像素。由于相邻的同尺寸、同比例的anchor是在feature map上的距离是一个点，对应到输入图片中就是16个像素。在一定程度上可以认为anchor的精度为16个像素。不过还需要考虑原图相比于输入图片又做过缩放（这也是dataset返回的scale参数的作用，这个的scale指的是原图和输入图片的缩放尺度，和上面的scale不一样）。
# 4 实现方案
其实上半年好几次都要用到Faster R-CNN，但是每回看到各种上万行，几万行代码，简直无从下手。而且直到 [@罗若天](https://www.zhihu.com/people/bbc1cc9eda99b0214ede09c620e76109)大神的[ruotianluo/pytorch-faster-rcnn](https://link.zhihu.com/?target=https%3A//github.com/ruotianluo/pytorch-faster-rcnn) 之前，PyTorch的Faster R-CNN并未有合格的实现（速度和精度）。最早PyTorch实现的Faster R-CNN有[longcw/faster_rcnn_pytorch](https://link.zhihu.com/?target=https%3A//github.com/longcw/faster_rcnn_pytorch) 和 [fmassa/fast_rcnn](https://link.zhihu.com/?target=https%3A//github.com/pytorch/examples/tree/d8d378c31d2766009db400ac03f41dd837a56c2a/fast_rcnn) 后者是当之无愧的最简实现（1,245行代码，包括空行注释，纯Python实现），然而速度太慢，效果较差，fmassa最后也放弃了这个项目。前者又太过复杂，mAP也比论文中差一点（0.661VS 0.699）。当前github上的大多数实现都是基于py-faster-rcnn，RBG大神的代码很健壮，考虑的很全面，支持很丰富，基本上git clone下来，准备一下数据模型就能直接跑起来。然而对我来说太过复杂，我的脑细胞比较少，上百个文件，动不动就好几层的嵌套封装，很容易令人头大。趁着最近时间充裕了一些，我决定从头撸一个，刚开始写没多久，就发现[chainercv](https://link.zhihu.com/?target=https%3A//github.com/chainer/chainercv)内置了Faster R-CNN的实现，而且Faster R-CNN中用到的许多函数（比如对bbox的各种操作计算），chainercv都提供了内置支持(其实py-faster-rcnn也有封装好的函数，但是chainercv的文档写的太详细了！)。所以大多数函数都是直接copy&paste，把chainer的代码改成pytorch/numpy，增加了一些可视化代码等。不过cupy的内容并没有改成THTensor。因为cupy现在已经是一个独立的包，感觉比cffi好用（虽然我并不会C....）。

最终写了一个简单版本的Faster R-CNN，代码地址在 [github：simple-faster-rcnn-pytorch](https://link.zhihu.com/?target=https%3A//github.com/chenyuntc/simple-faster-rcnn-pytorch)
这个实现主要有以下几个特点：
* 代码简单：除去空行，注释，说明等，大概有2000行左右代码，如果想学习如何实现Faster R-CNN，这是个不错的参考。
* 效果够好：超过论文中的指标（论文mAP是69.9， 本程序利用caffe版本VGG16最低能达到0.70，最高能达到0.712，预训练的模型在github中提供链接可以下载）
* 速度足够快：TITAN Xp上最快只要3小时左右（关闭验证与可视化）就能完成训练
* 显存占用较小：3G左右的显存占用
## ^_^
这个项目其实写代码没花太多时间，大多数时间花在调试上。有报错的bug都很容易解决，最怕的是逻辑bug，只能一句句检查，或者在ipdb中一步一步的执行，看输出是否和预期一样，还不一定找得出来。不过通过一步步执行，感觉对Faster R-CNN的细节理解也更深了。

写完这个代码，也算是基本掌握了Faster R-CNN。在写代码中踩了许多坑，也学到了很多，其中几个收获/教训是：
* 在复现别人的代码的时候，不要自作聪明做什么“改进”，先严格的按照论文或者官方代码实现（比如把SGD优化器换成Adam，基本训不动，后来调了一下发现要把学习率降10倍，但是效果依旧远不如SGD）。
* 不要偷懒，尽可能的“Match Everything”。由于torchvision中有预训练好的VGG16，而caffe预训练VGG要求输入图片像素在0-255之间（torchvision是0-1），BGR格式的，标准化只减均值，不除以标准差，看起来有点别扭（总之就是要多写几十行代码+专门下载模型）。然后我就用torchvision的预训练模型初始化，最后用了一大堆的trick，各种手动调参，才把mAP调到0.7（正常跑，不调参的话大概在0.692附近）。某天晚上抱着试试的心态，睡前把VGG的模型改成caffe的，第二天早上起来一看轻轻松松0.705 ...
* 有个小trick：把别人用其它框架训练好的模型权重转换成自己框架的，然后计算在验证集的分数，如果分数相差无几，那么说明，相关的代码没有bug，就不用花太多时间检查这部分代码了。
* 认真。那几天常常一连几个小时盯着屏幕，眼睛疼，很多单词敲错了没发现，有些报错了很容易发现，但是有些就。。。 比如计算分数的代码就写错了一个单词。然后我自己看模型的泛化效果不错，但就是分数特别低，我还把模型训练部分的代码又过了好几遍。。。
* 纸上得来终觉浅, 绝知此事要coding。
* 当初要是再仔细读一读 [最近一点微小的工作](https://zhuanlan.zhihu.com/p/28455306)和[ruotianluo/pytorch-faster-rcnn](https://link.zhihu.com/?target=https%3A//github.com/ruotianluo/pytorch-faster-rcnn) 的readme，能少踩不少坑。

P.S. 在github上搜索faster rcnn，感觉有一半以上都是华人写的。
最后，求Star [github: simple-faster-rcnn-pytorch](https://link.zhihu.com/?target=https%3A//github.com/chenyuntc/simple-faster-rcnn-pytorch)


