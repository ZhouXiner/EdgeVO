//
// Created by zhouxin on 2020/4/27.
//
#include "EdgeVO/BackEnd.h"

namespace EdgeVO{
    BackEnd::BackEnd(SystemConfig::Ptr system, EdgeVO::CameraConfig::Ptr camera) {
        CameraConfig_ = camera;
        BackEndConfig_ = system;
    }

    void BackEnd::AddNewestFrame(Frame::Ptr &frame) {
        std::lock_guard<std::mutex> lock(KeyFrameMutex_);
        AllFrames_.push_back(frame);
        ++AllFrameNum_;
    }

    void BackEnd::AddNewestKeyFrame(Frame::Ptr &newestKF) {
        if(newestKF == nullptr){
            return;
        }
        ///First add frame
        AddNewestFrame(newestKF);
        ///Add the newestFrame
        {
            std::lock_guard<std::mutex> lock(KeyFrameMutex_);
            newestKF->KFId_ = AllKeyFrameNum_;
            AllKeyFrames_.push_back(newestKF);
            ++AllKeyFrameNum_;
        }
    }

    ///clean the local mapper, loop closer, and some data
    bool BackEnd::Reset() {
        ///remove the last N KeyFrame
        ///std::lock_guard<std::mutex> l(KeyFrameMutex_);
        return true;
    }

    std::vector<SE3> BackEnd::GetTryInitiPose() {
        std::vector<SE3> TryPose;
        SE3 identity;

        TryPose.push_back(identity);

        std::lock_guard<std::mutex> lock(KeyFrameMutex_);
        if(AllFrameNum_ > 2){
            const auto frame_b1 = AllFrames_[AllFrameNum_ - 1];
            const auto frame_b2 = AllFrames_[AllFrameNum_ - 2];
            const SE3 pose_b1_kf = frame_b1->Tcw_ * AllKeyFrames_[AllKeyFrameNum_ - 1]->Tcw_.inverse();
            const SE3 pose_b1_b2 = frame_b1->Tcw_ * frame_b2->Tcw_.inverse();
            TryPose.push_back(pose_b1_kf);
            TryPose.push_back(pose_b1_b2 * pose_b1_kf);
            TryPose.push_back(SE3::exp(pose_b1_b2.log() * 0.5)*pose_b1_kf);
            TryPose.push_back(pose_b1_b2 * pose_b1_b2 * pose_b1_kf);
            TryPose.push_back(pose_b1_b2 * pose_b1_b2 * pose_b1_b2 * pose_b1_kf);
        }
        return TryPose;
    }

    Frame::Ptr BackEnd::GetNewestKeyFrame() {
        if(AllKeyFrameNum_ == 0){
            LOG(WARNING) << "No KeyFrame Before!";
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(KeyFrameMutex_);
        return AllKeyFrames_[AllKeyFrameNum_ - 1];
    }

    std::vector<Frame::Ptr> BackEnd::ReturnKeyFrames() {
        return AllFrames_;
    }
}





