排版问题，另请参考：[https://shimo.im/docs/qJU5h6aN8UAsQVzH/](https://shimo.im/docs/qJU5h6aN8UAsQVzH/) 《Faster R-CNN论文及源码解读》，可复制链接后用石墨文档 App 打开

# 参考



R-CNN是目标检测领域中十分经典的方法，相比于传统的手工特征，R-CNN将卷积神经网络引入，用于提取深度特征，后接一个分类器判决搜索区域是否包含目标及其置信度，取得了较为准确的检测结果。Fast R-CNN和Faster R-CNN是R-CNN的升级版本，在准确率和实时性方面都得到了较大提升。在Fast R-CNN中，首先需要使用Selective Search的方法提取图像的候选目标区域(Proposal)。而新提出的Faster R-CNN模型则引入了RPN网络(Region Proposal Network)，将Proposal的提取部分嵌入到内部网络，实现了卷积层特征共享，Fast R-CNN则基于RPN提取的Proposal做进一步的分类判决和回归预测，因此，整个网络模型可以完成端到端的检测任务，而不需要先执行特定的候选框搜索算法，显著提升了算法模型的实时性。
# 模型概述
Faster R-CNN模型主要由两个模块组成：RPN候选框提取模块和Fast R-CNN检测模块，如下图所示，又可细分为4个部分：
* Conv Layer(backbone network)
* Region Proposal Network(RPN)
* RoI Pooling
* Classification and Regression

![图片](https://images-cdn.shimo.im/BOIwUmYXIxY8mLag/image.png!thumbnail)
![图片](https://img-blog.csdn.net/20180517221855856?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L29KaU1vRGVZZTEyMzQ1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
Faster R-CNN网络模型

* Conv Layer: backbone网络，卷积层包括一系列卷积(Conv + Relu)和池化(Pooling)操作，用于提取图像的特征(**feature maps**)，一般直接使用现有的经典网络模型(例如ZF、VGG16)，而且卷积层的权值参数为RPN和Fast RCNN所共享，这也是能够加快训练过程、提升模型实时性的关键所在。
* Region Proposal Network: RPN网络用于生成区域候选框**Proposal**，基于网络模型引入的多尺度Anchor，通过Softmax对anchors属于目标(foreground)还是背景(background)进行分类判决(二分类)，并使用Bounding Box Regression对anchors进行回归预测，获取Proposal的精确位置，并用于后续的目标识别与检测。
* RoI Pooling: 综合卷积层特征feature maps和候选框proposal的信息，将propopal在输入图像中的坐标映射到最后一层feature map(conv5-3)中，对feature map中的对应区域进行池化操作，得到固定大小(7×7)输出的池化结果，并与后面的全连接层相连。
* Classification and Regression: 全连接层后接两个子连接层——分类层(cls)和回归层(reg)，分类层用于判断Proposal的类别，回归层则通过bounding box regression预测Proposal的准确位置。

下图为Faster R-CNN测试网络结构(网络模型文件为faster_rcnn_test.pt)，可以清楚地看到图像在网络中的前向计算过程。对于一幅任意大小P×Q的图像，首先缩放至固定大小M×N(源码中是要求长边不超过1000，短边不超过600)，然后将缩放后的图像输入至采用VGG16模型的Conv Layer中，最后一个feature map为conv5-3，特征数(channels)为512。RPN网络在特征图conv5-3上执行3×3卷积操作，后接一个512维的全连接层，全连接层后接两个子连接层，分别用于anchors的分类和回归，再通过计算筛选得到proposals。RoIs Pooling层则利用Proposal从feature maps中提取Proposal feature进行池化操作，送入后续的Fast R-CNN网络做分类和回归。RPN网络和Fast R-CNN网络中均有分类和回归，但两者有所不同，**RPN中分类是判断conv5-3中对应的anchors属于目标和背景的概率(score)，并通过回归获取anchors的偏移和缩放尺度，根据目标得分值筛选用于后续检测识别的Proposal；Fast R-CNN是对RPN网络提取的Proposal做分类识别，并通过回归参数调整得到目标(Object)的精确位置**。具体的训练过程会在后面详述。接下来会重点介绍RPN网络和Fast R-CNN网络这两个模块，包括RPN网络中引入的Anchor机制、训练数据的生成、分类和回归的损失函数(Loss Function)计算以及RoI Pooling等。
![图片](https://images-cdn.shimo.im/NX11NxrEa8wpnIme/image.png!thumbnail)
Fast R-CNN test网络结构
# RPN
Region Proposal Network(RPN)
传统的目标检测方法中生成候选框都比较耗时，例如使用滑动窗口加图像金字塔的方式遍历图像，获取多尺度的候选区域；以及R-CNN、Fast R-CNN中均使用到的Selective Search的方法生成候选框。而Faster R-CNN则直接使用RPN网络，将检测框Proposal的提取嵌入到网络内部，通过共享卷积层参数的方式提升了Proposal的生成速度。
## Anchor
Anchor是RPN网络中一个较为重要的概念，传统的检测方法中为了能够得到多尺度的检测框，需要通过建立图像金字塔的方式，对图像或者滤波器(滑动窗口)进行多尺度采样。RPN网络则是使用一个3×3的卷积核，在最后一个特征图(conv5-3)上滑动，将卷积核中心对应位置映射回输入图像，生成3种尺度(scale){128^2,256^2,512^2}和3种长宽比(aspect ratio){1:1,1:2,2:1}共9种Anchor，如下图所示。特征图conv5-3每个位置都对应9个anchors，如果feature map的大小为W×H，则一共有W×H×9个anchors，滑动窗口的方式保证能够关联conv5-3的全部特征空间，最后在原图上得到多尺度多长宽比的anchors。
![图片](https://images-cdn.shimo.im/26iX0Bd8UHQJIj6T/image.png!thumbnail)
Anchor示意图
![图片](https://img-blog.csdn.net/2018051722195383?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L29KaU1vRGVZZTEyMzQ1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
原始图片上的 Anchor Centers
![图片](https://img-blog.csdn.net/20180517222013500?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L29KaU1vRGVZZTEyMzQ1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
左：Anchors；中：单个点的 Anchor；右：全部Anchors. 
最后一个feature map后面会接一个全连接层，如下图所示，全连接的维数和feature map的特征数(channels)相同。**对于原论文中采用的ZF模型，conv5的特征数为256，全连接层的维数也为256；对于VGG模型，conv5-3的特征数为512，全连接的的维数则为512，相当于feature map上的****每一个点****都输出一个512维的特征向量。**
![图片](https://images-cdn.shimo.im/KGZ2r4socU89yTgG/image.png!thumbnail)
RPN网络结构
### 关于anchors还有几点需要说明
* conv5-3上使用了3×3的卷积核，每个点都可以关联局部邻域的空间信息。
* conv5-3上**每个点前向映射得到k(k=9)个anchors，并且后向输出512维的特征向量**，而anchors的作用是分类和回归得到Proposal，因此全连接层后须接两个子连接层--分类层(cls)和回归层(reg)，分类层用于判断anchors属于目标还是背景，向量维数为2k；回归层用于计算anchors的偏移量和缩放量，共4个参数[dx,dy,dw,dh]，向量维数为4k。
* 关于bounding boxes：寻找图片中的边界框bounding boxes，边界框是具有不同尺寸sizes和长宽比aspect ratios 的矩形。假设，已经知道图片中有两个 objects，首先想到的是，训练一个网络，输出 8 个值：两对元组(xmin,ymin,xmax,ymax)，分别定义了每个 object 的边界框。这种方法存在一些基本问题。例如，当图片的尺寸和长宽比不一致时，良好训练模型来预测，会非常复杂。 另一个问题是无效预测：预测 xmin和 xmax时，需要保证 xmin<xmax。事实上，有一种更加简单的方法来预测 objects 的边界框，即，学习相对于参考boxes 的偏移量。假设参考box: (xcenter,ycenter,width,height)，待预测量 Δxcenter、Δycenter、Δwidth、Δheight，一般都是很小的值，以调整参考 box 更好的拟合所需要的。
* Anchors 是固定尺寸的边界框，是通过利用不同的尺寸和比例在图片上放置得到的 boxes，并作为第一次预测 object 位置的参考 boxes。
### anchor的补充说明
![图片](https://uploader.shimo.im/f/QqOWzzTUVNMWyyWC.png!thumbnail)
利用anchor是从第二列这个位置开始进行处理，这个时候，原始图片已经经过一系列卷积层和池化层以及relu，得到了这里的 feature：51x39x256（256是层数）。在这个特征参数的基础上，通过一个3x3的滑动窗口，在这个51x39的区域上进行滑动，stride=1，padding=2，这样一来，滑动得到的就是51x39个3x3的窗口。

**对于每个3x3的窗口，作者就计算这个滑动窗口的中心点所对应的原始图片的中心点。然后作者假定，这个3x3窗口，是从原始图片上通过SPP池化得到的，而这个池化的区域的面积以及比例，就是一个个的anchor。换句话说，对于每个3x3窗口，作者假定它来自9种不同原始区域的池化，但是这些池化在原始图片中的中心点，都完全一样。这个中心点，就是刚才提到的，3x3窗口中心点所对应的原始图片中的中心点。如此一来，在每个窗口位置，我们都可以根据9个不同长宽比例、不同面积的anchor，逆向推导出它所对应的原始图片中的一个区域，这个区域的尺寸以及坐标，都是已知的。而这个区域，就是我们想要的 proposal。所以我们通过滑动窗口和anchor，成功得到了 51x39x9 个原始图片的proposal。接下来，每个proposal我们只输出6个参数：每个 proposal 和 ground truth 进行比较得到的前景概率和背景概率(2个参数）（对应图上的 cls_score）；由于每个 proposal 和 ground truth 位置及尺寸上的差异，从 proposal 通过平移放缩得到 ground truth 需要的4个平移放缩参数（对应图上的 bbox_pred）。**
**所以根据我们刚才的计算，我们一共得到了多少个anchor box呢？  **
**51 x 39 x 9 = 17900 **
**约等于 20 k**

## 训练样本的生成
一般而言，特征图conv5-3的实际尺寸大致为60×40，那么一共可以生成60×40×9≈20k个anchors，显然不会将所有anchors用于训练，而是筛选一定数量的正负样本。对于数据集中包含有人工标定ground truth的图像，考虑一张图像上所有anchors:
* 首先过滤掉超出图像边界的anchors
* 对每个标定的ground truth，与其重叠比例IoU最大的anchor记为正样本，这样可以保证每个ground truth至少对应一个正样本anchor
* 对每个anchors，如果其与某个ground truth的重叠比例IoU大于0.7，则记为正样本(目标)；如果小于0.3，则记为负样本(背景)
* 再从已经得到的正负样本中随机选取256个anchors组成一个minibatch用于训练，而且正负样本的比例为1:1,；如果正样本不够，则补充一些负样本以满足256个anchors用于训练，反之亦然。
## Multi-task Loss Function
由于涉及到分类和回归，所以需要定义一个多任务损失函数(Multi-task Loss Function)，包括Softmax Classification Loss和Bounding Box Regression Loss，公式定义如下：
![图片](https://images-cdn.shimo.im/UVu0XBgW0WskGhoN/image.png!thumbnail)
**Softmax Classification**：对于RPN网络的分类层(cls)，其向量维数为2k = 18，考虑整个特征图conv5-3，则输出大小为W×H×18，正好对应conv5-3上每个点有9个anchors，而每个anchor又有两个score(fg/bg)输出，对于单个anchor训练样本，其实是一个二分类问题。为了便于Softmax分类，需要对分类层执行reshape操作，这也是由底层数据结构决定的。在caffe中，Blob的数据存储形式为Blob=[batch_size,channel,height,width]，而对于分类层(cls)，其在Blob中的实际存储形式为[1,2k,H,W]，而Softmax针对每个anchor进行二分类，所以需要在分类层后面增加一个reshape layer，将数据组织形式变换为[1,2,k∗H,W]，之后再reshape回原来的结构，caffe中有对softmax_loss_layer.cpp的reshape函数做如下解释：
```
"Number of labels must match number of predictions; "  
"e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "  
"label count (number of labels) must be N*H*W, "  
"with integer values in {0, 1, ..., C-1}.";
```
在上式中，pi为样本分类的概率值，p∗i为样本的标定值(label)，anchor为正样本时p∗i为1，为负样本时p∗i为0，Lcls为两种类别的对数损失(log loss)。

**Bounding Box Regression**：RPN网络的回归层输出向量的维数为4k = 36，回归参数为每个样本的坐标[x,y,w,h]，分别为box的中心位置和宽高，考虑三组参数预测框(predicted box)坐标[x,y,w,h]，anchor坐标[xa,ya,wa,ha]，ground truth坐标[x∗,y∗,w∗,h∗]，分别计算预测框相对anchor中心位置的偏移量以及宽高的缩放量{t}，ground truth相对anchor的偏移量和缩放量{t∗}
![图片](https://images-cdn.shimo.im/cJoeYEUeYyEuyvKX/image.png!thumbnail)
**回归目标就是让{t}尽可能地接近{t∗}**，所以回归真正预测输出的是{t}，而训练样本的标定真值为{t∗}。得到预测输出{t}后，通过上式(1)即可反推获取预测框的真实坐标。在损失函数中，回归损失采用Smooth L1函数
![图片](https://images-cdn.shimo.im/OJo2Xfxc6c8HtGAi/image.png!thumbnail)
Smooth L1损失函数曲线如下图所示，相比于L2损失函数，L1对离群点或异常值不敏感，可控制梯度的量级使训练更易收敛。
![图片](https://images-cdn.shimo.im/GDznomumff8ALeIK/image.png!thumbnail)
Smooth L1损失函数
在损失函数中，p∗iLreg这一项表示只有目标anchor(p∗i=1)才有回归损失，其他anchor不参与计算。这里需要注意的是，当样本bbox和ground truth比较接近时(IoU大于某一阈值)，可以认为上式的坐标变换是一种线性变换，因此可将样本用于训练线性回归模型，否则当bbox与ground truth离得较远时，就是非线性问题，用线性回归建模显然不合理，会导致模型不work。分类层(cls)和回归层(reg)的输出分别为{p}和{t}，两项损失函数分别由Ncls和Nreg以及一个平衡权重λ归一化。分类损失的归一化值为minibatch的大小，即Ncls=256；回归损失的归一化值为anchor位置的数量，即Nreg≈2400；λ一般取值为10，这样分类损失和回归损失差不多是等权重的。
## Proposal的生成
Proposal的生成就是将图像输入到RPN网络中进行一次前向(forward)计算，处理流程如下：
* 计算特征图conv5-3映射到输入图像的所有anchors，并通过RPN网络前向计算得到anchors的score输出和bbox回归参数
* 由anchors坐标和bbox回归参数计算得到预测框proposal的坐标
* 处理proposal坐标超出图像边界的情况(使得坐标最小值为0，最大值为宽或高)
* 滤除掉尺寸(宽高)小于给定阈值的proposal
* 对剩下的proposal按照目标得分(fg score)从大到小排序，提取前pre_nms_topN(e.g. 6000)个proposal
* 对提取的proposal进行非极大值抑制(non-maximum suppression,nms)，再根据nms后的foreground score，筛选前post_nms_topN(e.g. 300)个proposal作为最后的输出
# Fast R-CNN
对于RPN网络中生成的proposal，需要送入Fast R-CNN网络做进一步的精确分类和坐标回归，但proposal的尺寸可能大小不一，所以需要做RoI Pooling，输出统一尺寸的特征，再与后面的全连接层相连。
## RoI Pooling
对于传统的卷积神经网络，当网络训练好后输入图像的尺寸必须是固定值，同时网络输出的固定大小的向量或矩阵。如果输入图像大小不统一，则需要进行特殊处理，如下图所示：
* 从图像中crop一部分传入网络
* 将图像warp成需要的大小后传入网络

![图片](https://images-cdn.shimo.im/kYQL1UxKwZ8OCw4A/image.png!thumbnail)
crop与warp操作

可以从图中看出，crop操作破坏了图像的完整结构，warp操作破坏了图像的原始形状信息，引入了尺度误差和形变误差，两种方法的效果都不太理想。RPN网络生成的proposal也存在尺寸不一的情况，但论文中提出了RoI Pooling的方法解决这个问题。

RoI Pooling结合特征图conv5-3和proposal的信息，proposal在输入图像中的坐标[x1,y1,x2,y2]对应M×N尺度，将proposal的坐标映射到(M/16)×(N/16)大小的conv5-3中，然后将Proposal在conv5-3的对应区域水平和竖直均分为7等份，并对每一份进行Max Pooling或Average Pooling处理，得到固定大小(7×7)输出的池化结果，实现固定长度输出(fixed-length output)，如下图所示：
![图片](https://images-cdn.shimo.im/1U1zEjkdwdMWpj4R/image.png!thumbnail)
RoI Pooling示意图（1）

![图片](https://images-cdn.shimo.im/HyXtIQ2VNOQXfJiL/image.png!thumbnail)
RoI Pooling示意图（2）

RoI Pooling示意图（2）中左上角是一个特征映射图，右上角中蓝色的ROI与特征映射图重叠。为了使ROI生成目标尺寸，例如 2 x 2 ，于是将ROI分成4个大小相似或者相等的部分，每个部分取最大值，结果得到 2 x 2 的特征块，最后可以将它送入分类器和边界回归器中处理。 
![图片](https://github.com/deepsense-ai/roi-pooling/raw/master/roi_pooling_animation.gif)
RoI Pooling示意图（3）

## Classification and Regression
RoI Pooling层后接多个全连接层，最后为两个子连接层--分类层(cls)和回归层(reg)，如下图所示，和RPN的输出类似，只不过输出向量的维数不一样。如果类别数为N+1(包括背景)，分类层的向量维数为N+1，回归层的向量维数则为4(N+1)。还有一个关键问题是RPN网络输出的proposal如何组织成Fast R-CNN的训练样本：
* 对每个proposal，计算其与所有ground truth的重叠比例IoU
* 筛选出与每个proposal重叠比例最大的ground truth
* 如果proposal的最大IoU大于0.5则为目标(前景)，标签值(label)为对应ground truth的目标分类；如果IoU小于0.5且大于0.1则为背景，标签值为0
* 从2张图像中随机选取128个proposals组成一个minibatch，前景和背景的比例为1:3
* 计算样本proposal与对应ground truth的回归参数作为标定值，并且将回归参数从(4,)拓展为(4(N+1),)，只有对应类的标定值才为非0。

设定训练样本的回归权值，权值同样为4(N+1)维，且只有样本对应标签类的权值才为非0。
在源码实现中，用于训练Fast R-CNN的Proposal除了RPN网络生成的，还有图像的ground truth，这两者归并到一起，然后通过筛选组成minibatch用于迭代训练。Fast R-CNN的损失函数也与RPN类似，二分类变成了多分类，背景同样不参与回归损失计算，且只考虑proposal预测为标签类的回归损失。
![图片](https://images-cdn.shimo.im/EfHbAmXi6tMChbkQ/image.png!thumbnail)
Classification and Regression


# Faster R-CNN的训练
对于提取proposals的RPN，以及分类回归的Fast R-CNN，如何将这两个网络嵌入到同一个网络结构中，训练一个共享卷积层参数的多任务(Multi-task)网络模型。源码中有实现交替训练(Alternating training)和端到端训练(end-to-end)两种方式，这里介绍交替训练的方法：
* 训练RPN网络，用ImageNet模型M0初始化，训练得到模型M1
* 利用第一步训练的RPN网络模型M1，生成Proposal P1
* 使用上一步生成的Proposal，训练Fast R-CNN网络，同样用ImageNet模型初始化，训练得到模型M2
* 训练RPN网络，用Fast R-CNN网络M2初始化，且固定卷积层参数，只微调RPN网络独有的层，训练得到模型M3
* 利用上一步训练的RPN网络模型M3，生成Proposal P2
* 训练Fast R-CNN网络，用RPN网络模型M3初始化，且卷积层参数和RPN参数不变，只微调Fast R-CNN独有的网络层，得到最终模型M4

由训练流程可知，第4步训练RPN网络和第6步训练Fast R-CNN网络实现了卷积层参数共享。总体上看，训练过程只循环了2次，但每一步训练(M1，M2，M3，M4)都迭代了多次(e.g. 80k，60k)。对于固定卷积层参数，只需将学习率(learning rate)设置为0即可。

# 源码解析
以上关于RPN的训练，Proposal的生成，以及Fast R-CNN的训练做了的详细讲解，接下来结合网络模型图和部分源码，对这些模块做进一步的分析。

## train RPN
训练RPN的网络结构如下图所示，首先加载参数文件，并改动一些参数适应当前训练任务。在train_rpn函数中调用get_roidb、get_imdb、get_train_imdb_roidb等获取训练数据集，并通过调用gt_roidb和prepare_roidb方法对训练数据进行预处理，为样本增添一些属性，数据集roidb中的每个图像样本，主要有以下属性：
```
'image':图像存储路径
'width':图像宽
'height':图像高
'boxes':图像中bbox(groundtruth or proposal)的坐标[x1,y1,x2,y2]
'gt_classes':每个bbox对应的类索引(1~20)
'gt_overlaps':二维数组，shape=[num_boxes * num_classes]，每个bbox(ground truth)对应的类索引处取值为1，其余为0
'flipped':取值为True/False，用于标记有无将图像水平翻转
'seg_area':bbox的面积
'max_classes':bbox与所有ground truth的重叠比例IoU最大的类索引(gt_overlaps.argmax(axis=1))
'max_overlaps':bbox与所有ground truth的IoU最大值(gt_overlaps.max(axis=1))
```
![图片](https://images-cdn.shimo.im/VTt7nT0M724cyIdN/image.png!thumbnail)
train_rpn_model

获取数据集roidb中字典的属性后，设置输出路径output_dir，用来保存中间训练结果，然后调用train_net函数。在train_net函数中，首先调用filter_roidb，滤除掉既没有前景又没有背景的roidb。然后调用layer.py中的set_roidb方法，打乱训练样本roidb的顺序，将roidb中长宽比近似的图像放在一起。之后开始训练模型train_model，这里需要实例化每个层，对于第一层RoIDataLayer，通过setup方法进行实例化，并且在训练过程中通过forward方法，调用get_minibatch函数，获取每一次迭代训练的数据，在读取数据时，主要获取了3个属性组成Layer中的Blob
```
'data':单张图像数据im_blob=[1,3,H,W]
'gt_boxes':一幅图像中所有ground truth的坐标和类别[x1,y1,x2,y2,cls]
'im_info':图像的宽高和缩放比例 height,width,scale = [[im_blob.shape[2], im_blob.shape[2], im_scale[0]]]
```
从网络结构图中可以看出，input-data(RoIDataLayer)的下一层是rpn-data(AnchorTargetLayer)，rpn-data计算所有anchors与ground truth的重叠比例IoU，从中筛选出一定数量(256)的正负样本组成一个minibatch，用于RPN网络的训练，这一层的输出有如下属性：
```
'rpn_label':每个anchor对应的类别(1——fg，0——bg，-1——ignored)，shape=[1,1,A*height,width]
'rpn_bbox_targets':anchor与ground truth的回归参数[dx,dy,dw,dh]，shape=[1,A*4,height,width]
'rpn_box_inside_targets':回归损失函数中的样本权值，正样本为1，负样本为0，相当于损失函数中的p*，shape=[1,A*4,height,width]
'rpn_box_outside_targets':分类损失函数和回归损失函数的平衡权重，相当于λ，shape=[1,A*4,height,width]
注：height、width为特征图conv5-3的高宽，A=9为Anchor种数
```

对于分类损失rpn_loss_cls，输入的rpn_cls_scors_reshape和rpn_labels分别对应p与p∗；对于回归损失，输入的rpn_bbox_pred和rpn_bbox_targets分别对应{t}与{t∗}，pn_bbox_inside_weigths对应p∗，rpn_bbox_outside_weights对应λ。

## generate proposals
Proposal的生成只需将图像输入到RPN网络中，进行前向(forward)计算然后经过筛选即可得到，网络结构如下图所示
![图片](https://images-cdn.shimo.im/I2Mi8P116eYGq8Y9/image.png!thumbnail)
generate proposals
* rpn_proposals=imdb_proposals(rpn_net, imdb)开始。
* im = cv2.imread(imdb.image_path_at(i))读入图片数据。
* 调用 im_proposals生成单张图片的rpn proposals，以及得分。
* im_proposals函数会调用网络的forward方法，从而得到想要的boxes和scores。
* 最后将获取的proposal保存在python pickle文件中。
## train Fast R-CNN
训练Fast R-CNN的网络结构如下图所示，首先设置参数适应训练任务，在预处理数据时，调用的不再是gt_roidb方法，而是rpn_roidb，通过使用类imdb的静态方法merge_roidb，将rpn_roidb和gt_roidb归并为一个roidb，因此数据集中的’boxes’属性除了包含ground truth，还有RPN网络生成的proposal，可通过上一步保存的文件直接读取。通过add_bbox_regression_targets方法给roidb的样本增添了额外的属性’bbox_targets’，用于表示回归参数的标定值。属性’gt_overlaps’是所有proposal与ground truth通过计算IoU得到的。最后就是调用get_minibatch方法从2张图像中选取128个proposal作为一次迭代的训练样本，读取数据时，获取如下属性组成Layer中的Blob
```
'data':图像数据
'rois':proposals的坐标[batch_inds,x1,y1,x2,y2]
'label':proposals对应的类别(0~20)
'bbox_targets':proposal回归参数的标定值，shape = [128, 4(N+1)]
'box_inside_targets':回归损失函数中的样本权值，正样本为1，负样本为0，相当于损失函数中的p*
'rpn_box_outside_targets':分类损失函数和回归损失函数的平衡权重，相当于λ
```
![图片](https://images-cdn.shimo.im/waKMLaGKwlQK21pX/image.png!thumbnail)
train_fast_rcnn_model
损失函数的计算与RPN网络类似。在Faster R-CNN中，自定义的Python Layer包括RoIDataLayer、AnchorTargetLay、ProposalLayer，都只实现了前向计算forward，因为这些Layer的作用是获取用于训练网络的数据，而对网络本身没有贡献任何权值参数，也不传播梯度值，因此不需要实现反向传播backward。
# reference
* Paper: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
* Paper: R-CNN：Rich feature hierarchies for accurate object detection and semantic segmentation
* Paper: SPP-Net: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
* Paper: Fast R-CNN
* Code: Caffe implement of Faster RCNN
* Code: Tensorflow implement of Faster RCNN
* [http://blog.csdn.net/iamzhangzhuping/article/category/6230157](http://blog.csdn.net/iamzhangzhuping/article/category/6230157)
* [http://www.infocool.net/kb/Python/201611/209696.html](http://www.infocool.net/kb/Python/201611/209696.html)
* [http://www.cnblogs.com/venus024/p/5717766.html](http://www.cnblogs.com/venus024/p/5717766.html)
* [http://blog.csdn.net/zy1034092330/article/details/62044941](http://blog.csdn.net/zy1034092330/article/details/62044941)

