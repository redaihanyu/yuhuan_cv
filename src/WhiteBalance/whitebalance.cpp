//
//  whitebalance.cpp
//  cv_scripts
//
//  Created by 余欢 on 2018/12/20.
//  Copyright © 2018 Steven. All rights reserved.
//

#include "whitebalance.hpp"

Mat grayWorld(Mat &srcImg){
    vector<Mat> imageRGB;
    
    //RGB三通道分离
    split(srcImg, imageRGB);
    
    //求原始图像的RGB分量的均值
    double R, G, B;
    B = mean(imageRGB[0])[0];
    G = mean(imageRGB[1])[0];
    R = mean(imageRGB[2])[0];
    
    //需要调整的RGB分量的增益
    double KR, KG, KB;
    KB = (R + G + B) / (3 * B);
    KG = (R + G + B) / (3 * G);
    KR = (R + G + B) / (3 * R);
    
    //调整RGB三个通道各自的值
    imageRGB[0] = imageRGB[0] * KB;
    imageRGB[1] = imageRGB[1] * KG;
    imageRGB[2] = imageRGB[2] * KR;
    
    //RGB三通道图像合并
    Mat dst(srcImg.size(), srcImg.type()) ;
    merge(imageRGB, dst);
    return dst ;
}

Mat simplestColorBalance(Mat &srcImg, double satLevel, const float inputMin, const float inputMax, const float outputMin, const float outputMax) {
    // 在rgb三通道上分别计算直方图
    // 将1%的最大值和最小值设置为255和0
    // 其余值映射到(0, 255), 这样使得每个值通道的值在rgb中分布较均匀, 以实现简单的颜色平衡
    vector<Mat> splitImage;
    split(srcImg, splitImage) ;
    
    int depth = 2;      // depth of histogram tree
    int bins = 16;      // number of bins at each histogram level
    int total = splitImage[0].rows * splitImage[0].cols;
    int nElements = int(pow((float)bins, (float)depth));        // number of elements in histogram tree
    
    for (size_t k = 0; k < splitImage.size(); ++k){
        vector<int> hist(nElements, 0);
        uchar *pImag = splitImage[k].data;
        // histogram filling
        for (int i = 0; i < total; i++){
            int pos = 0;
            float minValue = inputMin - 0.5f;
            float maxValue = inputMax + 0.5f;
            float interval = float(maxValue - minValue) / bins;
            
            uchar val = pImag[i];
            for (int j = 0; j < depth; ++j){
                int currentBin = int((val - minValue + 1e-4f) / interval);
//                cout << pos << "  " << currentBin << "  " << (pos + currentBin)  << endl ;
                ++hist[pos + currentBin];
                
                pos = (pos + currentBin) * bins;
                minValue += currentBin * interval;
                interval /= bins;
            }
        }
        
        int p1 = 0, p2 = bins - 1;
        int n1 = 0, n2 = total;
        float minValue = inputMin - 0.5f;
        float maxValue = inputMax + 0.5f;
        float interval = float(maxValue - minValue) / bins;
        
        // searching for s1 and s2
        for (int j = 0; j < depth; ++j)
        {
            while (n1 + hist[p1] < satLevel * total)
            {
                n1 += hist[p1++];
                minValue += interval;
            }
            p1 *= bins;
            
            while (n2 - hist[p2] > (1 - satLevel) * total)
            {
                n2 -= hist[p2--];
                maxValue -= interval;
            }
            p2 = p2*bins - 1;
            
            interval /= bins;
        }
        
        splitImage[k] = (outputMax - outputMin) * (splitImage[k] - minValue) / (maxValue - minValue) + outputMin;
    }
            
    Mat dst(srcImg.size(), srcImg.type()) ;
    merge(splitImage, dst);
    
    return dst ;
}

Mat simpleWhiteBalance(Mat &srcImage, float qlow, float qhigh, const float amin, const float amax){
    vector<Mat> splitImage;
    split(srcImage, splitImage) ;
    
    int pix_size = splitImage[0].rows * splitImage[0].cols ;    // 图片的像素数量
    int lowcount = int(pix_size * qlow) ;   // 被置于0和255的像素个数
    int highcount = int(pix_size * qhigh) ;
    // 分通道处理
    for(size_t c = 0; c < splitImage.size(); ++c){
        uchar *pImage = splitImage[c].data ;

        // 像素值灰度值的最小值、最大值
        double minv = 0.0, maxv = 0.0;
        minMaxIdx(splitImage[c], &minv, &maxv);
        vector<int> bins(pix_size, 0) ;
        for(int element_index = 0; element_index < pix_size; element_index++){
            ++bins[int(pImage[element_index])] ;
        }
        
        // 初始化最小、最大的像素值
        int pix_low = 0, pix_high = 0, count_low = 0, count_high = 0 ;
        for(size_t bin_index = 0; bin_index < bins.size(); bin_index++){
            if(count_low >= lowcount){
                ;
            }else{
                count_low += bins[bin_index] ;
                pix_low = int(bin_index) ;
            }
            
            if(count_high >= highcount){
                ;
            }else{
                count_high += bins[bins.size() - bin_index - 1] ;
                pix_high = int(bins.size()) - int(bin_index) - 1 ;
            }
        }
        
        // 像素值变换
//        splitImage[c] = (splitImage[c] - pix_low) * 255 / (pix_high - pix_low) ;
        for(int element_index = 0; element_index < pix_size; element_index++){
            int p_val = int(pImage[element_index]) ;
            if(p_val <= pix_low){
               pImage[element_index]  = amin ;
            }else if(p_val >= pix_high){
                pImage[element_index] = amax ;
            }else{
                pImage[element_index] = int((p_val - pix_low) * 255 / (pix_high - pix_low)) ;
            }

        }

    }
    
    Mat dst(srcImage.size(), srcImage.type()) ;
    merge(splitImage, dst) ;
    
    return dst ;
    
}
