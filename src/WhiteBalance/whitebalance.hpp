//
//  whitebalance.hpp
//  cv_scripts
//
//  Created by 余欢 on 2018/12/20.
//  Copyright © 2018 Steven. All rights reserved.
//

#ifndef whitebalance_hpp
#define whitebalance_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std ;
using namespace cv ;

Mat grayWorld(Mat &srcImg) ;
Mat simplestColorBalance(Mat &srcImg, double satLevel, const float inputMin = 0.0f, const float inputMax = 255.0f,
                         const float outputMin = 0.0f, const float outputMax = 255.0f) ;
Mat simpleWhiteBalance(Mat &srcImage, float qlow = 0.01f, float qhigh = 0.01f, const float amin=0.0f, const float amax=255.0f) ;

#endif /* whitebalance_hpp */
