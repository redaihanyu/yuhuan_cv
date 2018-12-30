# 1、概述
这个算法被应用于Adobe Photoshop的"auto lever"命令，其认为一张色彩均衡的图片，最亮的应该是白色，最暗的应该是黑色。因此，我们可以通过伸缩R、G、B各个通道的直方图使其横跨0-255整个区间，从而可以消除色偏。

为了处理极端值，该方法提升一定比例的饱和度使亮的像素变成白色，使暗的像素变成黑色。提升的饱和度比例是一个会影响输出质量的可调节参数，通常设置为0.01。

简单的说就是：在rgb三通道上分别统计每个像素值的出现次数。将1%的最大值和最小值设置为255和0。其余值映射到0-255，这样使得每个值通道的值在rgb中分布较均匀。达到颜色平衡的结果。
# 2、原理介绍
自动对比度调整的方法中，我们假设alow和ahight为单前图像中的最小值和最大值，需要映射的范围为[amin,amax]，为了使得图像映射到整个映射范围，我们首先把最小值alow映射到0，之后用比例因子(amax-amin)/(ahigh-alow)增加其对比度，随后加上amin使得计算出来的值映射到需要的映射范围。其具体公式为：
![图片](https://uploader.shimo.im/f/YZilFcKtoKoIh1U4.png!thumbnail)
对于8bit图像而言，amin = 0,amax = 255。故上述公式可以改写为：
![图片](https://uploader.shimo.im/f/GjwT2cT179wIlE4v.png!thumbnail)
实现原理图如下：
![图片](https://uploader.shimo.im/f/RkUpaxdsejEVjaIc.png!thumbnail)
实际的图像中，上述的映射函数容易受到个别极暗或极亮像素的影响，导致映射可能出现错误。为了避免这种错误，我们选取较低像素和较亮像素的一定比例qlow，qhigh，并根据此比例重新计算alow和ahigh，得到vlow和vhigh，以vlow和vhigh为原始的图像的亮度范围进行映射。我们可以通过累计直方图H(i)很方便的计算出vlow和vhigh。
![图片](https://uploader.shimo.im/f/xE3858ZODBkBsKhA.png!thumbnail)
其中，0<=qlow ,qhigh<=1,qlow+qhigh<=1,M*N为图像的像素数量。
其原理图如下所示：
![图片](https://uploader.shimo.im/f/akkbtcQWdhQqH4SJ.png!thumbnail)
注：图中alow`为Vlow，ahigh`为Vhigh
# 3、实现方法
1. **排序：**为了获取高亮度一定比例像素和较低亮度的一定比例像素，需要对整个图像的灰度值进行排序，方便选择对应的像素。
2. **选取：**从排序的像素数组中选取一定比例的高亮和较暗像素。假设s = s1 + s2 = [0,100]，我们需要选取N*S / 100个像素。其中Vmin和Vmax应该选择位于排序后数组的位置为N*S1   /100和N*(1 - S2/100)处的像素灰度值。
3. **填充：**由上一步定义的Vmin和Vmax，把原始图像中低于或等于Vlow的像素灰度值赋予0；把原始像素中高于或等于Vmax的像素灰度值赋予255。
4. **映射：**使用上述的映射函数把原始图像的像素映射到[0,255]范围内。

当图像较大的时候，图像的像素可达到百万级，对这么大数量的像素进行排序，往往效率比较低。另外一种方法可以像上述自动对比度调整那样，建立一个256大小的数组，再以数组中N * S1/100和N(1 - S2 / 100)处的像素灰度值赋予amin和amax。然后再进行像素值得映射。

伪代码如下：
![图片](https://uploader.shimo.im/f/eDY6Q0q6pvU85pE1.png!thumbnail)

# 4、reference
[http://stanford.edu/~sujason/ColorBalancing/simplestcb.html](http://stanford.edu/~sujason/ColorBalancing/simplestcb.html)

