//
// Created by zhouxin on 2020/3/24.
//
#include "EdgeVO/Inputs.h"

namespace EdgeVO{
    FrameDataset::FrameDataset(const SystemConfig::Ptr system_config,const CameraConfig::Ptr camera) {
        ImageConfig_ = system_config;
        CameraConfig_ = camera;
    }

    /**Using the thread to read Images*/
    void FrameDataset::Run() {
        IsActive_ = true;
        std::thread th(&FrameDataset::ReadImages,this);
        th.detach();
    }

    bool FrameDataset::IsActive() {
        std::lock_guard<std::mutex> lock(FrameQueueLock_);
        return IsActive_ || !FramesList_.empty();
    }

    void FrameDataset::ReadImages() {
        FileLists_.open(ImageConfig_->ImageFilePath_);
        if(!FileLists_.is_open()){
            LOG(ERROR) << "Bad File Path!";
            exit(0);
        }

        /**Read all the RGB and depth Images*/
        std::string InputLine,RGBInputFile,DepthInputFile;
        double RGBTimeStamp,DepthTimeStamp;
        ImgCount_ = 0;
        while(getline(FileLists_,InputLine)){
            if(InputLine[0] == '#' || InputLine.empty()) continue;
            std::istringstream single_associate(InputLine);
            single_associate >> RGBTimeStamp >> RGBInputFile >> DepthTimeStamp >> DepthInputFile;

            cv::Mat depth,rgb;
            rgb = cv::imread(ImageConfig_->ImageDatasetPath_ + RGBInputFile,1);
            depth = cv::imread(ImageConfig_->ImageDatasetPath_ + DepthInputFile,CV_LOAD_IMAGE_UNCHANGED);

            depth.convertTo(depth,CV_32FC1,1.0f / ImageConfig_->ImageDepthScale_);

            Frame::Ptr frame(std::make_shared<Frame>(ImageConfig_,CameraConfig_));

            frame->Initialize(rgb,depth,RGBTimeStamp,DepthTimeStamp);
            {
                std::lock_guard<std::mutex> lock(FrameQueueLock_);

                FramesList_.push_back(frame);
                //std::cout << "After: " << FramesQueue_.size() << std::endl;
                ++ImgCount_;
            }
        }
        LOG(INFO) << "Dataset has " << ImgCount_ << " Pictures";
        IsActive_ = false;
    }

    void FrameDataset::GetBound(cv::Mat& dtImg,cv::Mat& depthImg) {
        if(ImageConfig_->FernUseDepth_){
            double DepthMinBound,DepthMaxBound;
            minMaxLoc(depthImg,&DepthMinBound,&DepthMaxBound,0,0);
            //std::cout << " max depth: " << DepthMaxBound << std::endl;
            if(static_cast<float>(DepthMinBound) < DepthBound_[0]) DepthBound_[0] = static_cast<float>(DepthMinBound);
            if(static_cast<float>(DepthMaxBound) > DepthBound_[1]) DepthBound_[1] = static_cast<float>(DepthMaxBound);
        }
        if(ImageConfig_->FernUseDT_){
            double DTMinBound,DTMaxBound;
            minMaxLoc(dtImg,&DTMinBound,&DTMaxBound,0,0);

            if(static_cast<float>(DTMinBound) < DTBound_[0]) DTBound_[0] = static_cast<float>(DTMinBound);
            if(static_cast<float>(DTMaxBound) > DTBound_[1]) DTBound_[1] = static_cast<float>(DTMaxBound);
        }
    }
    Frame::Ptr FrameDataset::GetNewestFrame() {
        std::lock_guard<std::mutex> lock(FrameQueueLock_);
        if(FramesList_.empty()){
            return nullptr;
        }

        auto frame = FramesList_.front();
        FramesList_.pop_front();
        return frame;
    }
}
