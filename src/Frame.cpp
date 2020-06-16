//
// Created by zhouxin on 2020/3/24.
//

#include "EdgeVO/Frame.h"

namespace EdgeVO{
    Frame::Frame() {}

    Frame::Frame(const SystemConfig::Ptr system_config,const CameraConfig::Ptr camera) {
        FrameConfig_ = system_config;
        CameraConfig_ = camera;
    }

    void Frame::Initialize(const cv::Mat &rgbImg, const cv::Mat &depthImg, const double rgbStamp, const double depthStamp) {
        RGBInputImg_ = rgbImg;
        RGBTimeStamp_  = rgbStamp;
        DepthInputImg_ = depthImg;
        DepthTimeStamp_ = depthStamp;

        static long long int factoryId = 0;
        Id_ = factoryId++;

        GetPyramidImgs(); /**Images Pyramid*/
        GetPyramidDTInfo(); /**Dt Info Pyramid*/
        GetEdgePixels(); /**EdgePixel Pyramid*/
        GetNN(); /**Get nearest edge*/
        DeBugImg();
    }


    void Frame::GetEdge(cv::Mat &EdgeImg, const cv::Mat &GrayImg) {
        GaussianBlur(GrayImg,GrayImg,cv::Size(3,3),2);
        cv::Canny(GrayImg,EdgeImg,FrameConfig_->CannyThresholdL_,FrameConfig_->CannyThresholdH_,3);
    }

    void Frame::GetDT(cv::Mat &DTImg, const cv::Mat &EdgeImg) {
        cv::distanceTransform(255 - EdgeImg,DTImg,CV_DIST_L2,CV_DIST_MASK_PRECISE);
    }

    void Frame::GetPyramidImgs() {
        for(size_t lvl = 0;lvl < CameraConfig_->PyramidLevel_;++lvl){
            RGBImgs_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_8UC3));
            DepthImgs_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_32FC1));
            GrayImgs_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_8UC1));
            EdgeImgs_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],cv::DataType<uint8_t>::type));
            DTImgs_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_32FC1));
        }

        RGBInputImg_.copyTo(RGBImgs_[0]);
        DepthInputImg_.copyTo(DepthImgs_[0]);

        cv::cvtColor(RGBImgs_[0],GrayImgs_[0],CV_BGR2GRAY);
        GetEdge(EdgeImgs_[0],GrayImgs_[0]);
        GetDT(DTImgs_[0],EdgeImgs_[0]);

        for(size_t lvl = 1;lvl < CameraConfig_->PyramidLevel_;++lvl){
            /**cv::Size中先列后行，而直接创建则是先行后列*/
            cv::resize(RGBImgs_[lvl -1],RGBImgs_[lvl],cv::Size(CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl]));
            cv::resize(DepthImgs_[lvl -1],DepthImgs_[lvl],cv::Size(CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl]));

            cv::cvtColor(RGBImgs_[lvl],GrayImgs_[lvl],CV_BGR2GRAY);
            GetEdge(EdgeImgs_[lvl],GrayImgs_[lvl]);
            GetDT(DTImgs_[lvl],EdgeImgs_[lvl]);
        }
    }

    void Frame::GetPyramidDTInfo() {
        PyramidDT_.resize(CameraConfig_->PyramidLevel_);

        for(size_t lvl = 0; lvl < CameraConfig_->PyramidLevel_;lvl++){
            PyramidDT_[lvl] = (reinterpret_cast<Vec3*>(Eigen::internal::aligned_malloc(CameraConfig_->SizeW_[lvl]*CameraConfig_->SizeH_[lvl]*sizeof(Vec3))));
        }

        for(size_t lvl = 0; lvl < CameraConfig_->PyramidLevel_;lvl++){
            int Lvlw = CameraConfig_->SizeW_[lvl];
            int Lvlh = CameraConfig_->SizeH_[lvl];

            Vec3* PyramidDTLvl = PyramidDT_[lvl];

            for(int col = 0;col < Lvlw;++col){
                for(int row = 0;row < Lvlh;++row){
                    auto idx = col + row*Lvlw;

                    PyramidDTLvl[idx][0] = DTImgs_[lvl].at<float>(row,col);

                    if(Utility::InBorder(col,row,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],1)){
                        /**sobel*/
                        //PyramidDTLvl[idx][1] = (-DTImgs_[lvl].at<float>(row - 1,col + 1) - 2*DTImgs_[lvl].at<float>(row ,col + 1) - DTImgs_[lvl].at<float>(row + 1,col + 1)
                        //        + DTImgs_[lvl].at<float>(row - 1,col - 1) + 2*DTImgs_[lvl].at<float>(row,col - 1) + DTImgs_[lvl].at<float>(row + 1,col - 1)) / 8;

                        //PyramidDTLvl[idx][2] = (-DTImgs_[lvl].at<float>(row + 1,col - 1) - 2*DTImgs_[lvl].at<float>(row + 1,col) - DTImgs_[lvl].at<float>(row + 1,col + 1)
                        //                     + DTImgs_[lvl].at<float>(row - 1,col - 1) + 2*DTImgs_[lvl].at<float>(row - 1,col) + DTImgs_[lvl].at<float>(row - 1,col + 1)) / 8;
                        ///测试灰度梯度
                        PyramidDTLvl[idx][1] = static_cast<float>((-GrayImgs_[lvl].at<uint8_t>(row - 1,col + 1) - 2*GrayImgs_[lvl].at<uint8_t>(row ,col + 1) - GrayImgs_[lvl].at<uint8_t>(row + 1,col + 1)
                                                + GrayImgs_[lvl].at<uint8_t>(row - 1,col - 1) + 2*GrayImgs_[lvl].at<uint8_t>(row,col - 1) + GrayImgs_[lvl].at<uint8_t>(row + 1,col - 1)) / 8.0f);

                        PyramidDTLvl[idx][2]  = static_cast<float>((-GrayImgs_[lvl].at<uint8_t>(row + 1,col - 1) - 2*GrayImgs_[lvl].at<uint8_t>(row + 1,col) - GrayImgs_[lvl].at<uint8_t>(row + 1,col + 1)
                                                + GrayImgs_[lvl].at<uint8_t>(row - 1,col - 1) + 2*GrayImgs_[lvl].at<uint8_t>(row - 1,col) + GrayImgs_[lvl].at<uint8_t>(row - 1,col + 1)) / 8.0f);
                        //std::cout << PyramidDTLvl[idx][1] << " " << PyramidDTLvl[idx][2] << std::endl;
                    }
                }
            }
        }

    }

    void Frame::GetEdgePixels() {
        EdgePixels_.resize(CameraConfig_->PyramidLevel_);
        EdgeNumCountVec_.resize(CameraConfig_->PyramidLevel_);

        for(size_t lvl = 0; lvl < CameraConfig_->PyramidLevel_;++lvl){
            int patch_w = static_cast<int>((CameraConfig_->SizeW_[lvl] + 0.5) / FrameConfig_->EdgePatchSize_);
            int patch_h = static_cast<int>((CameraConfig_->SizeH_[lvl] + 0.5) / FrameConfig_->EdgePatchSize_);
            EdgeNumCountVec_[lvl].resize(FrameConfig_->EdgePatchSize_ * FrameConfig_->EdgePatchSize_);
            EdgeNumCountVec_[lvl].assign(EdgeNumCountVec_[lvl].size(),0);
            PatchNumEachLvl_[lvl] = FrameConfig_->EdgePatchSize_ * FrameConfig_->EdgePatchSize_;
        }

        for(size_t lvl = 0;lvl < CameraConfig_->PyramidLevel_;++lvl){
            int Lvlw = CameraConfig_->SizeW_[lvl];
            int Lvlh = CameraConfig_->SizeH_[lvl];
            int patch_w = static_cast<int>((CameraConfig_->SizeW_[lvl]) / FrameConfig_->EdgePatchSize_);
            int patch_h = static_cast<int>((CameraConfig_->SizeH_[lvl]) / FrameConfig_->EdgePatchSize_);
            int num_each_patch = static_cast<int>(FrameConfig_->TrackMostNum_[lvl] / PatchNumEachLvl_[lvl]);

            for(size_t col = FrameConfig_->ImageBound_; col < Lvlw - FrameConfig_->ImageBound_;++col){
                for(size_t row = FrameConfig_->ImageBound_; row < Lvlh - FrameConfig_->ImageBound_;++row){
                    int w_posi = static_cast<int>(col /  patch_w);
                    int z_posi = static_cast<int>(row /  patch_h);
                    int patch_locate = z_posi * FrameConfig_->EdgePatchSize_ + w_posi;
                    if((EdgeNumCountVec_[lvl][patch_locate] > num_each_patch || patch_locate > FrameConfig_->EdgePatchSize_ * FrameConfig_->EdgePatchSize_) && FrameConfig_->UseEdgePatch_){
                        continue;
                    }

                    const uint8_t EdgePixelVal = EdgeImgs_[lvl].at<uint8_t>(row,col);
                    if(EdgePixelVal == 1 || EdgePixelVal == 255){
                        const float DepthVal = DepthImgs_[lvl].at<float>(row,col);
                        const float DTVal = DTImgs_[lvl].at<float>(row,col);
                        if(DepthVal > FrameConfig_->ImageDepthMin_ && DepthVal < FrameConfig_->ImageDepthMax_){
                            EdgePixels_[lvl].push_back(std::make_shared<EdgePixel>(col,row,lvl,DepthVal,DTVal)); /**record like (x,y),也就是说如果要定位到图像中，还得是(y,x) the weight and color not added yet*/
                            ++EdgeNumCountVec_[lvl][patch_locate];
                        }
                    }
                }
            }
        }
    }


    void Frame::GetNN() {
        for(size_t lvl = 0;lvl < CameraConfig_->PyramidLevel_;++lvl){
            LocationX_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_32SC1));
            LocationY_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_32SC1));
        }

        for(size_t lvl = 0;lvl < CameraConfig_->PyramidLevel_;++lvl) {
            int Lvlw = CameraConfig_->SizeW_[lvl];
            int Lvlh = CameraConfig_->SizeH_[lvl];
            for(int col = 0;col < Lvlw;++col) {
                for (int row = 0; row < Lvlh; ++row) {
                    if(DTImgs_[lvl].at<float>(row,col) == 0){
                        LocationX_[lvl].at<int>(row,col) = col;
                        LocationY_[lvl].at<int>(row,col) = row;
                        continue;
                    }
                    bool flag = false;
                    int maskSize = 1;
                    float min_dis = 999999;
                    int minDTLocation[2] = {0};
                    while(true){
                        int x_min = std::max(0,col - maskSize);
                        int x_max = std::min(col + maskSize,Lvlw-1);
                        int y_min = std::max(0,row - maskSize);
                        int y_max = std::min(row + maskSize,Lvlh-1);

                        ///分为四部分
                        for(int y=y_min;y<=y_max;++y){
                            if(DTImgs_[lvl].at<float>(y,x_min) == 0){
                                float dis = Utility::calcEuclideanDistance(x_min,y,col,row);
                                if(dis < min_dis){
                                    minDTLocation[0] = x_min;
                                    minDTLocation[1] = y;
                                    min_dis = dis;
                                    flag = true;
                                }
                            }
                        }
                        if(min_dis == maskSize){
                            LocationX_[lvl].at<int>(row,col) = minDTLocation[0];
                            LocationY_[lvl].at<int>(row,col) = minDTLocation[1];
                            break;
                        }

                        for(int y=y_min;y<=y_max;++y){
                            if(DTImgs_[lvl].at<float>(y,x_max) == 0){
                                float dis = Utility::calcEuclideanDistance(x_max,y,col,row);
                                if(dis < min_dis){
                                    minDTLocation[0] = x_max;
                                    minDTLocation[1] = y;
                                    min_dis = dis;
                                    flag = true;
                                }
                            }
                        }
                        if(min_dis == maskSize){
                            LocationX_[lvl].at<int>(row,col) = minDTLocation[0];
                            LocationY_[lvl].at<int>(row,col) = minDTLocation[1];
                            break;
                        }

                        for(int x = x_min;x<=x_max;++x){
                            if(DTImgs_[lvl].at<float>(y_min,x) == 0){
                                float dis = Utility::calcEuclideanDistance(x,y_min,col,row);
                                if(dis < min_dis){
                                    minDTLocation[0] = x;
                                    minDTLocation[1] = y_min;
                                    min_dis = dis;
                                    flag = true;
                                }
                            }
                        }
                        if(min_dis == maskSize){
                            LocationX_[lvl].at<int>(row,col) = minDTLocation[0];
                            LocationY_[lvl].at<int>(row,col) = minDTLocation[1];
                            break;
                        }

                        for(int x = x_min;x<=x_max;++x){
                            if(DTImgs_[lvl].at<float>(y_max,x) == 0){
                                float dis = Utility::calcEuclideanDistance(x,y_max,col,row);
                                if(dis < min_dis){
                                    minDTLocation[0] = x;
                                    minDTLocation[1] = y_max;
                                    min_dis = dis;
                                    flag = true;
                                }
                            }
                        }
                        if(min_dis == maskSize){
                            LocationX_[lvl].at<int>(row,col) = minDTLocation[0];
                            LocationY_[lvl].at<int>(row,col) = minDTLocation[1];
                            break;
                        }

                        if(flag){
                            LocationX_[lvl].at<int>(row,col) = minDTLocation[0];
                            LocationY_[lvl].at<int>(row,col) = minDTLocation[1];
                            break;
                        }
                        ++maskSize;
                    }
                }
            }
        }

    }
    void Frame::GetLocation() {
        for(size_t lvl = 0;lvl < CameraConfig_->PyramidLevel_;++lvl){
            LocationX_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_32SC1));
            LocationY_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_32SC1));
        }

        int maskSize = FrameConfig_->LocationMaskSize_;

        for(size_t lvl = 0;lvl < CameraConfig_->PyramidLevel_;++lvl){
            int Lvlw = CameraConfig_->SizeW_[lvl];
            int Lvlh = CameraConfig_->SizeH_[lvl];
            for(int col = 0;col < Lvlw;++col) {
                for (int row = 0; row < Lvlh; ++row) {
                    if(DTImgs_[lvl].at<float>(row,col) == 0){
                        LocationX_[lvl].at<int>(row,col) = col;
                        LocationY_[lvl].at<int>(row,col) = row;
                        continue;
                    }
                    float minDT_ = 99999;
                    int minDTLocation[2] = {0};
                    for(int x = std::max(0,col - maskSize);x<=std::min(col + maskSize,Lvlw-1);++x){
                        for(int y = std::max(0,row - maskSize);y<=std::min(row + maskSize,Lvlh-1);++y){
                            //std::cout << x << "  " << y << std::endl;
                            const float DTVal = DTImgs_[lvl].at<float>(y,x);
                            if(DTVal < minDT_){
                                minDTLocation[0] = x;
                                minDTLocation[1] = y;
                                minDT_ = DTVal;
                        }
                        }
                    }
                    LocationX_[lvl].at<int>(row,col) = minDTLocation[0];
                    LocationY_[lvl].at<int>(row,col) = minDTLocation[1];
                }
            }
        }
    }

    Vec2 Frame::GetNearestEdge(int x, int y, int lvl) {
        ///just like the 并查集
        Vec2 output;
        //bool flag = (x == 38 && y == 480) ? true : false;
        //std::cout << "Start: " << x << "x" << y << std::endl;
        while(true){
            const int last_x = LocationX_[lvl].at<int>(y,x);
            const int last_y = LocationY_[lvl].at<int>(y,x);
            //if(flag) std::cout << last_x << " " << last_y << " " << DTImgs_[lvl].at<float>(last_y,last_x) << std::endl;
            //std::cout << "next: " << int(last_x) << "x" << int(last_y) << std::endl;
            if(int(last_x) == x && int(last_y) == y){
                output[0] = x;
                output[1] = y;
                //std::cout << "\n" << std::endl;
                return output;
            }else{
                x = last_x;
                y = last_y;
            }
        }
    }

    Vec2 Frame::GetGradient(const Vec2 &p, int lvl) {
        int x = static_cast<int>(p[0]);
        int y = static_cast<int>(p[1]);
        Vec3 baseVec = PyramidDT_[lvl][x + y*CameraConfig_->SizeW_[lvl]];
        return Vec2{baseVec[1],baseVec[2]};
    }


    void Frame::DeBugImg() {
        //check the DT Info

        /*
        for(size_t i = 0;i < FrameConfig_->PyramidLevel_;i++){
            cv::imshow("DT_" + std::to_string(i),DTImgs_[i]);
        }
         */

        cv::Mat rgb = RGBImgs_[0];

        for(auto& pixel:EdgePixels_[0]){
            int x = pixel->Hostx_ + rand() % 100;
            int y = pixel->Hosty_ + rand() % 100;
            //if(x < 350 || y > 300) continue;
            //std::cout << "id: " << Id_ << std::endl;
            //std::cout << x << " " << y << std::endl;
            if(!Utility::InBorder(x,y,CameraConfig_->SizeW_[0],CameraConfig_->SizeH_[0],1)) continue;
            cv::circle(rgb,cv::Point(x,y),1,cv::Scalar(0,0,255));
            //if(baseVec[1] == baseVec[2] == 0){
            //    cv::circle(rgb,cv::Point(x,y),1,cv::Scalar(0,0,255));
            //}

            /*
            Vec2 posi;
            while(true){
                const int last_x = LocationX_[0].at<int>(y,x);
                const int last_y = LocationY_[0].at<int>(y,x);
                //std::cout << last_x << " " << last_y << std::endl;
                //if(flag) std::cout << last_x << " " << last_y << " " << DTImgs_[lvl].at<float>(last_y,last_x) << std::endl;
                //std::cout << "next: " << int(last_x) << "x" << int(last_y) << std::endl;
                if(int(last_x) == x && int(last_y) == y){
                    posi[0] = x;
                    posi[1] = y;
                    //cv::circle(rgb,cv::Point(posi[0],posi[1]),1,cv::Scalar(255,0,0));
                    //std::cout << "\n" << std::endl;
                    break;
                }else{
                    //cv::circle(rgb,cv::Point(last_x,last_y),1,cv::Scalar(0,255,0));
                    //cv::line(rgb,cv::Point(x,y),cv::Point(last_x,last_y),cv::Scalar(0,255,0));
                    x = last_x;
                    y = last_y;
                }
            }
             */

            Vec2 posi = GetNearestEdge(x,y,0);
            //if(!Utility::InBorder(posi,CameraConfig_->SizeW_[0],CameraConfig_->SizeH_[0],1)) continue;
            cv::circle(rgb,cv::Point(x,y),1,cv::Scalar(0,0,255));
            cv::circle(rgb,cv::Point(posi[0],posi[1]),1,cv::Scalar(0,255,0));
            cv::line(rgb,cv::Point(x,y),cv::Point(posi[0],posi[1]),cv::Scalar(255,0,0));
        }

        cv::resize(rgb,rgb,cv::Size(CameraConfig_->SizeW_[0]*2,CameraConfig_->SizeH_[0]*2));
        //cv::imshow("rgb" + std::to_string(Id_),rgb);
        std::string path = "/home/zhouxin/Desktop/NNImage/size1_step/";
        cv::imwrite(path + "rgb" + std::to_string(Id_) + ".jpg",rgb);

        //check the RGB Info
/*
        for(size_t i = 0;i < FrameConfig_->PyramidLevel_;i++){
            cv::imshow("RGB_" + std::to_string(i),RGBImgs_[i]);
        }
*/
        //check the depth info
/*
        for(size_t lvl = 0;lvl < FrameConfig_->PyramidLevel_;lvl++){
            double MinBound,MaxBound;
            minMaxLoc(DepthImgs_[lvl],&MinBound,&MaxBound,0,0);
            int Lvlw = CameraConfig_->SizeW_[lvl];
            int Lvlh = CameraConfig_->SizeH_[lvl];
            std::cout << "lvl: " << lvl << " Min depth: " << MinBound << " Max depth: " << MaxBound << std::endl;
            cv::Mat depth_img = DepthImgs_[lvl].clone();
            for(int col = 0;col < Lvlw;++col){
                for(int row = 0;row < Lvlh;++row){
                    if(depth_img.at<float>(row,col) > FrameConfig_->ImageDepthMax_ || depth_img.at<float>(row,col) < FrameConfig_->ImageDepthMin_){
                        depth_img.at<float>(row,col) = 0;
                    } else{
                        depth_img.at<float>(row,col) = (depth_img.at<float>(row,col) - MinBound) / MaxBound;
                    }
                }
            }
            cv::imshow("Depth_" + std::to_string(lvl),depth_img);
        }
*/
/*
        for(size_t lvl = 0;lvl < FrameConfig_->PyramidLevel_;lvl++){
            cv::Mat rgd_img = RGBImgs_[lvl];
            int patch_w = static_cast<int>((CameraConfig_->SizeW_[lvl] + 0.5) / FrameConfig_->EdgePatchSize_);
            int patch_h = static_cast<int>((CameraConfig_->SizeH_[lvl] + 0.5) / FrameConfig_->EdgePatchSize_);
            for(int i = 0;i<CameraConfig_->SizeW_[lvl];i = i + patch_w){
                cv::line(rgd_img,cv::Point(i,0),cv::Point(i,CameraConfig_->SizeH_[lvl]),cv::Scalar(0,255,0));
            }

            for(int i = 0;i<CameraConfig_->SizeH_[lvl];i = i + patch_h){
                cv::line(rgd_img,cv::Point(0,i),cv::Point(CameraConfig_->SizeW_[lvl],i),cv::Scalar(255,0,0));
            }

            for(auto& pixel : EdgePixels_[lvl]){
                Vec2 position = pixel->ReturnPixelPosition();
                cv::circle(rgd_img, cv::Point2f(position[0],position[1]), 0.5, cv::Scalar(0, 0, 255),
                           2);
            }
            std::cout << "lvl: " << lvl << " size: " << EdgePixels_[lvl].size() << std::endl;
            cv::imshow("RGB_" + std::to_string(lvl),rgd_img);
        }
*/
        //cv::waitKey(1);

    }


    int Frame::ReturnPixelSize() {
        auto size = 0;
        for(auto &eachpyramidpixel : EdgePixels_){
            size = size + eachpyramidpixel.size();
        }
        return size;
    }

    void Frame::UpdateHostId(int id) {
        HostId_ = id;
    }

    void Frame::UpdatePoseC1C2(const SE3 &pose) {
        Tth_ = pose;
    }
    void Frame::UpdatePoseCW(const SE3 &pose) {
        Tcw_ = pose;
    }

    void Frame::UpdateRefKF(EdgeVO::Frame::Ptr &kf) {
        RefKF_ = kf;
    }
    SE3 Frame::ReturnPoseCW() {
        return Tcw_;
    }

    SE3 Frame::ReturnPoseWC() {
        return Tcw_.inverse();
    }
    SE3 Frame::ReturnPoseC1C2() {
        return Tth_;
    }

    int Frame::ReturnId() {
        return Id_;
    }
    int Frame::ReturnHostId() {
        return HostId_;
    }

    Frame::Ptr Frame::ReturnRefKF() {
        return RefKF_;
    }

    int Frame::ReturnGoodSize() {
        int goodNum = 0;
        for(auto &pixel:EdgePixels_[0]){
            if(pixel->EdgeStatus_ == EdgePixel::EdgeStatus::Good){
                ++goodNum;
            }
        }
        return goodNum;
    }
    int Frame::ReturnBadSize() {
        int Badsize = 0;
        for(auto& pixel:EdgePixels_[0]){
            if(pixel->EdgeStatus_ == EdgePixel::EdgeStatus::Marged || pixel->EdgeStatus_ == EdgePixel::EdgeStatus::Fail)
                Badsize++;
        }
        return Badsize;
    }

    int Frame::ReturnBadAndUncertainSize() {
        return ReturnUncertainSize() + ReturnBadSize();
    }

    int Frame::ReturnUncertainSize() {
        int Uncertainsize = 0;
        for(auto& pixel:EdgePixels_[0]){
            if(pixel->UsedNum_ == 0) Uncertainsize++;
        }
        return Uncertainsize;
    }
}






















