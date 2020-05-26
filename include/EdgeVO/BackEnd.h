//
// Created by zhouxin on 2020/5/26.
//

#ifndef EDGEVO_BACKEND_H
#define EDGEVO_BACKEND_H
#include "CommonInclude.h"
#include "Camera.h"
#include "Config.h"
#include "Frame.h"


///combine the local mapper, loop and graph optimization here
namespace EdgeVO{
    class BackEnd{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<BackEnd> Ptr;

        BackEnd(SystemConfig::Ptr system,CameraConfig::Ptr camera);

        ///input part
        void AddNewestKeyFrame(Frame::Ptr& newestKF,bool checkloop = true);
        void AddNewestFrame(Frame::Ptr& frame);

        ///reset
        bool Reset();

        ///output part
        Frame::Ptr GetNewestKeyFrame();
        std::vector<Frame::Ptr> ReturnKeyFrames();
        std::vector<SE3> GetTryInitiPose();

        ///graph optimization
        CameraConfig::Ptr CameraConfig_;
        SystemConfig::Ptr BackEndConfig_;

        std::mutex FrameMutex_;
        std::mutex KeyFrameMutex_;

        std::vector<EdgeVO::Frame::Ptr> AllFrames_;
        std::vector<EdgeVO::Frame::Ptr> AllKeyFrames_;

        int AllKeyFrameNum_ = 0;
        int AllFrameNum_ = 0;
    };
}
#endif //EDGEVO_BACKEND_H
