//
//  main.cpp
//  cv_scripts
//
//  Created by 余欢 on 2018/12/20.
//  Copyright © 2018 Steven. All rights reserved.
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "whitebalance.hpp"

using namespace cv;
using namespace std;

int main(){
    Mat imageSource = imread("/Users/yuhuan/Desktop/白平衡测试图片/004.jpg");
    imshow("imgSrc", imageSource);
    
    // 白平衡：灰度世界法
    imshow("img_gray_world", grayWorld(imageSource)) ;
    // 白平衡：简单白平衡(实现1)
    imshow("img_simple_balance", simplestColorBalance(imageSource, 0.01f)) ;
    // 白平衡：简单白平衡(实现2)
    imshow("img_simple_balance_2", simpleWhiteBalance(imageSource, 0.01f, 0.01f)) ;
    
    waitKey();
    return 0;
}
