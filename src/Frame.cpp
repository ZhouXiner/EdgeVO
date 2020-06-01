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
        //DeBugImg();
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
                        PyramidDTLvl[idx][1] = (-DTImgs_[lvl].at<float>(row - 1,col + 1) - 2*DTImgs_[lvl].at<float>(row ,col + 1) - DTImgs_[lvl].at<float>(row + 1,col + 1)
                                + DTImgs_[lvl].at<float>(row - 1,col - 1) + 2*DTImgs_[lvl].at<float>(row,col - 1) + DTImgs_[lvl].at<float>(row + 1,col - 1)) / 8;

                        PyramidDTLvl[idx][2] = (-DTImgs_[lvl].at<float>(row + 1,col - 1) - 2*DTImgs_[lvl].at<float>(row + 1,col) - DTImgs_[lvl].at<float>(row + 1,col + 1)
                                             + DTImgs_[lvl].at<float>(row - 1,col - 1) + 2*DTImgs_[lvl].at<float>(row - 1,col) + DTImgs_[lvl].at<float>(row - 1,col + 1)) / 8;
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

    void Frame::GetLocation() {
        for(size_t lvl = 0;lvl < CameraConfig_->PyramidLevel_;++lvl){
            LocationX_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_8UC1));
            LocationY_.push_back(cv::Mat(CameraConfig_->SizeH_[lvl],CameraConfig_->SizeW_[lvl],CV_8UC1));
        }

        int maskSize = FrameConfig_->LocationMaskSize_;
        int minDTLocation[2] = {0};
        for(size_t lvl = 0;lvl < CameraConfig_->PyramidLevel_;++lvl){
            int Lvlw = CameraConfig_->SizeW_[lvl];
            int Lvlh = CameraConfig_->SizeH_[lvl];
            for(int col = 0;col < Lvlw;++col) {
                for (int row = 0; row < Lvlh; ++row) {
                    float minDT_ = 99999;
                    for(int x = std::max(0,col - maskSize);x<std::min(col + maskSize,Lvlw);++x){
                        for(int y = std::max(0,row - maskSize);y<std::min(row + maskSize,Lvlw);++y){
                            if(DTImgs_[lvl].at<float>(row,col) < minDT_){
                                minDTLocation[0] = x;
                                minDTLocation[1] = y;
                                minDT_ = DTImgs_[lvl].at<float>(row,col);
                            }
                        }
                    }
                    LocationX_[lvl].at<uint8_t>(row,col) = minDTLocation[0];
                    LocationY_[lvl].at<uint8_t>(row,col) = minDTLocation[1];
                }
            }
        }
    }

    Vec2 Frame::GetNearestEdge(int x, int y, int lvl) {
        ///just like the 并查集
        Vec2 output;
        while(true){
            const uint8_t last_x = LocationX_[lvl].at<uint8_t>(y,x);
            const uint8_t last_y = LocationX_[lvl].at<uint8_t>(y,x);
            if(last_x == x && last_y == y){
                output[0] = x;
                output[1] = y;
                return output;
            }else{
                x = last_x;
                y = last_y;
            }
        }
    }

    void Frame::DeBugImg() {
        //check the DT Info

        for(size_t i = 0;i < FrameConfig_->PyramidLevel_;i++){
            cv::imshow("DT_" + std::to_string(i),DTImgs_[i]);
        }


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

        cv::waitKey(0);

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






















