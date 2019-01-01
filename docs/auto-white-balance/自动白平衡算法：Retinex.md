* **如果遇到排版问题，另请参考**[https://shimo.im/docs/gYFi1bxQExMDMXkZ/](https://shimo.im/docs/gYFi1bxQExMDMXkZ/) 《自动白平衡算法：White Patch Retinex》，可复制链接后用石墨文档 App 打开
* c++代码请参考src目录
* python 完整代码在当前目录
# 1、reference
* [https://blog.csdn.net/ajianyingxiaoqinghan/article/details/71435098](https://blog.csdn.net/ajianyingxiaoqinghan/article/details/71435098)
* [http://blog.csdn.net/carson2005/article/details/9502053](http://blog.csdn.net/carson2005/article/details/9502053) 
* [https://www.cnblogs.com/sleepwalker/p/3676600.html](https://www.cnblogs.com/sleepwalker/p/3676600.html)
* [https://hk.saowen.com/a/a12e6cb05f8a6962b6822d0cf613f59d874e21ea988b1c4c8e5fefcef7a3e82d](https://hk.saowen.com/a/a12e6cb05f8a6962b6822d0cf613f59d874e21ea988b1c4c8e5fefcef7a3e82d)
[一种颜色保持的彩色图像增强新算法.pdf](https://uploader.shimo.im/f/37OLur0honYgJwks.pdf)


# 2、算法原理
人眼对物体颜色的感知，在外界照度条件发生变化时，仍能保持相对不变，表现出色彩常性(色彩恒常性)。Retinex是一种常用的建立在科学实验和科学分析基础上的图像增强方法，它是Edwin.H.Land于1963年提出的。就跟Matlab是由Matrix和Laboratory合成的一样，Retinex这个词是由视网膜(Retina)和大脑皮层(Cortex)两个词组合构成的。Land之所以设计这个词，是为了表明他不清楚视觉系统的特性究竟取决于此两个生理结构中的哪一个，抑或是与两者都有关系。Land人为，颜色恒常直觉不受照明变化的影响，只与视觉系统对物体的反射性质的知觉有关。Land的retinex模式是建立在以下三个假设之上的：
* 真实世界是无颜色的，我们所感知的颜色是光与物质的相互作用的结果。我们见到的水是无色的，但是水膜—肥皂膜却是显现五彩缤纷，那是薄膜表面光干涉的结果。
* 每一颜色区域由给定波长的红、绿、蓝三原色构成的；
* 三原色决定了每个单位区域的颜色。

**Retinex理论的基础理论是物体的颜色是由物体对长波（红色）、中波（绿色）、短波（蓝色）光线的反射能力来决定的，而不是由反射光强度的绝对值来决定的，物体的色彩不受光照非均匀性的影响，具有一致性，即retinex是以色感一致性（颜色恒常性）为基础的。**

不同于传统的线性、非线性的只能增强图像某一类特征的方法，Retinex可以在动态范围压缩、边缘增强和颜色恒常三个方面达到平衡，因此可以对各种不同类型的图像进行自适应的增强。
* **动态范围压缩**是指Retinex算法删除了图像成像过程中由于曝光过度和曝光不足而产生的低亮度和高亮度的强度值。
* **颜色常性**是指Retinex算法处理过的图像能消除照明对成像颜色的影响。

40多年来，研究人员模仿人类视觉系统发展了Retinex算法，从单尺度Retinex算法，改进成多尺度加权平均的MSR算法，再发展成彩色恢复多尺度MSRCR算法。
# 3、单尺度SSR(Single Scale Retinex)
## 3.1 原理
一幅给定的图像S(x,y)可以分解为两个不同的图像：反射图像R(x,y)和入射图像（也有人称之为亮度图像）L(x,y)，其原理图如下所示： 
![图片](https://img-blog.csdn.net/20170508211020962?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWppYW55aW5neGlhb3FpbmdoYW4=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
如上图所示，图像可以看做是入射图像和反射图像构成，入射光照射在反射物体上，通过反射物体的反射，形成反射光进入人眼，也就是人类所能看到的图像。
![图片](https://uploader.shimo.im/f/Ac5IyVUXzdMsL4zI.png!thumbnail)
公式（1）
其中，L(x, y)表示入射光图像，它直接决定了图像中像素所能达到的动态范围，被认为是图像S(x, y)的低频分量，可以用一个低通滤波器来过滤，我们应该尽量去除；R(x, y)表示反射性质图像，它反映图像的内在属性，我们应该最大程度的保留，S(x, y)表示人眼所能接收的反射光图像。

基于Retinex的图像增强的目的就是从原始图像S中估计出光照L，从而分解出R，消除光照不均的影响，以改善图像的视觉效果，正如人类视觉系统那样。在处理中，通常将图像转至对数域，即![图片](https://uploader.shimo.im/f/ilCf27r2rhIXM6cD.png!thumbnail)，从而将乘积关系转换为和的关系，而且对数形式接近人眼亮度感知能力：
	![图片](https://uploader.shimo.im/f/6MiORJyDxn01rkI5.png!thumbnail)
Retinex方法的核心就是估测照度L，从图像S中估测L分量，并去除L分量，得到原始反射分量R，即：
	![图片](https://uploader.shimo.im/f/sQI2fRpYY5U8Mxfc.png!thumbnail)
函数f(x)实现对照度L的估计（实际很多都是直接估计r分量）。

Retinex认为在原始图像中，通过某种方法去除或者降低入射光图像的影响，从而尽量的保留物体本质的反射图像。从数学的角度来说，求解R(x, y)是一个奇异问题，只能通过数学方法近似的估计计算。根据入射图像估计方法的不同，先后涌现出很多的Retinex算法，虽然各种方法的表现形式不一样，但是实质基本是一致的，其一般的处理流程如下：
![图片](https://uploader.shimo.im/f/Oikd0rOoS0IeYpsz.png!thumbnail)
一般的，我们**把****照****射图像假设估****计为****空间平滑图像**，原始图像为S(x, y)，反射图像为R(x, y)，亮度图像为L(x, y)，可以得出上面的公式(1)，以及下面的公式：
![图片](https://uploader.shimo.im/f/Jn4MeVAVBR8i6p6Y.png!thumbnail)
公式（2）

![图片](https://uploader.shimo.im/f/AuR4ZWaUCv4Pw0Cy.png!thumbnail)
公式（3）
这里，r(x, y)是输出图像，式(3)中后面中括号里的运算是卷积运算。F(x, y)是中心/围绕函数，表示为：
![图片](https://uploader.shimo.im/f/jtm1HVTJ0jsNnlPV.png!thumbnail)
公式（4）
式(4)中的C是高斯环绕尺度，它的取值必须满足下式：
![图片](https://uploader.shimo.im/f/otLokIJQkL0VnU1N.png!thumbnail)
公式（5）
上面的式中可以看出，SSR算法中的卷积是对输入图像的计算，其物理意义是通过计算像素点与周围区域在加权平均的作用下，估计图像中亮度的变化，并将L(x,y)去除，只保留S(x,y)属性。
## 3.2 实现流程
单尺度Retinex算法SSR的实现流程可以概括如下：

1. 读原图S(x, y)： 
  * 若原图为灰度图：将图像各像素的灰度值由整数型(int)转换为浮点数(float)，并转换到对数域；
  * 若原图为彩色图：将颜色分通道处理，每个分量像素值由整数型(int)转换为浮点数(float)，并转换到对数域；
2. 输入高斯环绕尺度C，把积分运算离散化，转为求和运算，通过上式(4)(5)确定λ的值；
3. 由式(3)得r(x, y)； 
  * 若原图是灰度图，则只有一个r(x, y)；
  * 若原图为彩色图，则每个通道都有一个对应的r(x, y)；
4. 将r(x, y)从对数域转换到实数域，得到输出图像R(x, y)；
5. 此时的R(x, y)值的范围并不是0–255，所以还需要进行线性拉伸并转换成相应的格式输出显示。

前面的公式中，中心环绕函数F(x, y)用的是低通函数，这样能够在算法中估计出入射图像对应原始图像的低频部分。从原始图像中除去低频照射部分，就会留下原始图像所对应的高频分量。高频分量很有价值，因为在人类的视觉系统中，人眼对边缘部分的高频信息相当敏感，所以SSR算法可以较好的增强图像中的边缘信息。 

## 3.3 物理意义与不足
SSR的物理意义是，中心/围绕函数相当于一个低通滤波器，在对数空间中，将原图像减去中心/围绕函数与原图像的卷积的值，实际上原图像被减去了平滑的部分。

中心/围绕函数的尺度参数C的选取不同会直接影响到被处理图像的颜色再现，C越小，动态范围压缩的能力越强，亮度较暗区域(如阴影)的细节能得到较好的增强，但由于平均对比度范围较小，输出Retinex会产生颜色失真；反之，C越大，颜色保真度越高。为了平衡两种增强效果，就必须选择一个较为恰当的高斯尺度常量C。C值一般取值在80–100之间。

因此，Retinex算法不能完全摆脱光照条件变化的影响。
## 3.4 代码
python 代码
```
def restore(img):
    # 恢复到0-255阈值
    for i in range(img.shape[2]):
        img_i_min = np.min(img[:, :, i])
        img_i_max = np.max(img[:, :, i])
        img[:, :, i] = (img[:, :, i] - img_i_min) / (img_i_max - img_i_min) * 255

    return np.uint8(img)


# 单尺度Retinex，sigma = 300
def singleScaleRetinex(img, sigma):
    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(_temp == 0, 0.01, _temp)  # 避免出现0值
    img_ssr = np.log10(img + 0.01) - np.log10(gaussian)  # 按照公式计算
    
```
##     return restore(img_ssr)

3.5 结果
![图片](https://uploader.shimo.im/f/yJg9BvAToRo6CR4M.png!thumbnail)![图片](https://uploader.shimo.im/f/HpraSbwODlsrdTT8.png!thumbnail)
sigma == 300
# 4、**多尺度MSR(Multi-Scale Retinex)**
## 4.1 原理和流程
MSR是在SSR基础上发展来的，优点是可以同时保持图像高保真度与对图像的动态范围进行压缩的同时，MSR也可实现色彩增强、颜色恒常性、局部动态范围压缩、全局动态范围压缩，也可以用于X光图像增强。 

基于SSR算法，MSR算法描述如下：
1. 需要对原始图像每个进行高斯模糊，得到模糊后的图像Li(x, y)，其中i表示尺度数；
2. 在每个尺度下进行累加计算![图片](https://uploader.shimo.im/f/3rK8wjUuXR40ANuI.png!thumbnail)

式中，K是高斯中心环绕函数的个数。当K=1时，MSR退化为SSR。 其中wi表示每个尺度的权重，要求各尺度权重之和必须等于1，经典的取值为等值权重。通常来讲，为了保证兼有SSR高、中、低三个尺度的优点来考虑，K取值通常为3，且有：w1=w2=w3=1/3
## 4.2 结果
![图片](https://uploader.shimo.im/f/Gf6FctJoyXoOGNm6.png!thumbnail)
## 4.3 代码
```
def restore(img):
    # 恢复到0-255阈值
    for i in range(img.shape[2]):
        img_i_min = np.min(img[:, :, i])
        img_i_max = np.max(img[:, :, i])
        img[:, :, i] = (img[:, :, i] - img_i_min) / (img_i_max - img_i_min) * 255

    return np.uint8(img)

def singleScaleRetinex(img, sigma):
    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(_temp == 0, 0.01, _temp)    # 避免出现0值
    img_ssr = np.log10(img + 0.01) - np.log10(gaussian)     # 按照公式计算

    return restore(img_ssr)

# 多尺度Retinex，sigma_list = [15, 80, 250]
def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img * 1.0)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)
```
##     retinex /= len(sigma_list)

    return restore(retinex)

4.3 不足
一般的Retinex算法对光照图像估计时，都会假设初始光照图像是缓慢变化的，即光照图像是平滑的。但实际并非如此，亮度相差很大区域的边缘处，图像光照变化并不平滑。所以在这种情况下，Retinex增强算法在亮度差异大区域的增强图像会产生光晕。 

另外MSR常见的缺点还有边缘锐化不足，阴影边界突兀，部分颜色发生扭曲，纹理不清晰，高光区域细节没有得到明显改善，对高光区域敏感度小等。如下图：
![图片](https://uploader.shimo.im/f/2JykJnd7sqQ86kIF.png!thumbnail)![图片](https://uploader.shimo.im/f/nucX7McjkkQaxkWL.png!thumbnail)
# 5、带颜色恢复的MSR方法MSRCR(Multi-Scale Retinex with Color Restoration)
## 5.1 原理
在前面的增强过程中，图像可能会因为增加了噪声，而使得图像的局部细节色彩失真，不能显现出物体的真正颜色，整体视觉效果变差。针对这一点不足，MSRCR在MSR的基础上，加入了色彩恢复因子C来调节由于图像局部区域对比度增强而导致颜色失真的缺陷。 
改进算法如下所示：
![图片](https://uploader.shimo.im/f/Jmiik2yIsbg6pqKG.png!thumbnail)
![图片](https://uploader.shimo.im/f/I3h6UOiCpSkuEVzX.png!thumbnail)![图片](https://uploader.shimo.im/f/dDRLDtsAIXwQlg0s.png!thumbnail)
其中参数说明如下：
* Ii(x, y)表示第i个通道的图像
* Ci表示第i个通道的彩色回复因子，用来调节3个通道颜色的比例；
* f(·)表示颜色空间的映射函数；
* β是增益常数，一般取值46；
* α是受控制的非线性强度，一般取值125；

MSRCR算法利用彩色恢复因子C，调节原始图像中3个颜色通道之间的比例关系，从而把相对较暗区域的信息凸显出来，达到了消除图像色彩失真的缺陷。 
处理后的图像局部对比度提高，亮度与真实场景相似，在人们视觉感知下，图像显得更加逼真。

但是MSRCR算法处理图像后，像素值一般会出现负值。所以从对数域r(x, y)转换为实数域R(x, y)后，需要通过改变增益Gain，偏差Offset对图像进行修正。使用公式可以表示为： 
![图片](https://uploader.shimo.im/f/dD6qMtoHLE0J6fTX.png!thumbnail)
式(11)中，
* G表示增益Gain，一般取值5
* O表示偏差Offset，一般取值25
## 5.2 结果
![图片](https://uploader.shimo.im/f/FA0jnXyt66s2ItJw.png!thumbnail)
## 5.3 代码
```
import cv2
import numpy as np

# 恢复到0-255阈值
def restore(img):
    for i in range(img.shape[2]):
        img_i_min = np.min(img[:, :, i])
        img_i_max = np.max(img[:, :, i])
        img[:, :, i] = (img[:, :, i] - img_i_min) / (img_i_max - img_i_min) * 255

    return np.uint8(img)


# 单尺度Retinex，sigma = 300
def singleScaleRetinex(img, sigma):
    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(_temp == 0, 0.01, _temp)  # 避免出现0值
    img_ssr = np.log10(img + 0.01) - np.log10(gaussian)  # 按照公式计算

    return restore(img_ssr)


# 多尺度Retinex，sigma_list = [15, 80, 250]
def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img * 1.0)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)
    retinex /= len(sigma_list)

    return restore(retinex)


def colorRestoration(img, alpha, beta):
    img = np.float32(img)
    img_sum = np.sum(img, axis=2, keepdims=True)
    return beta * (np.log10(img * alpha) - np.log10(img_sum))


def msrcr(img, sigma_list, alpha, beta, G, b):
    img_msr = multiScaleRetinex(img, sigma_list)
    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * img_color * img_msr + b

    return restore(img_msrcr)
