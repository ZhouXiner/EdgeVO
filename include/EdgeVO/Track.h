//
// Created by zhouxin on 2020/5/26.
//

#ifndef EDGEVO_TRACK_H
#define EDGEVO_TRACK_H
#include <ceres/ceres.h>
#include <cmath>
#include <set>

#include "../Utils/LGSX.h"
#include "../Utils/Utility.h"
#include "EdgeVO/CommonInclude.h"
#include "EdgeVO/Config.h"
#include "EdgeVO/Frame.h"
#include "EdgeVO/Camera.h"

namespace lsd_slam{class LGS6; }
/**Track between two corresponding Frames, initlize the Pose between two Frames*/
namespace EdgeVO{

    class DeBugInfo{
    public:
        double MinError_ = 999;
        double MaxError_ = -1;
        double AllBeforeError_ = 0;
        double AllAfterError_ = 0;
        int GoodNum_ = 0;
        int LostNumAfterCeres_ = 0;
        int LostNumAfterCheck_ = 0;
        int KeyNum_ = 0;
        int FrameCount_ = 0;
    };

    class BufferInfo{
    public:
        void Reset(){
            GoodEdgesNum_ = 0;
            BadEdgesNum_ = 0;
            SumError_ = 0;
            ReprojectError_ = 0;
            ReprojectErrorSelf_ = 0;
            DTError_ = 0;
        }
        int GoodEdgesNum_ = 0;
        int BadEdgesNum_ = 0;
        double SumError_ = 0;
        double DTError_;
        double ReprojectError_ = 0;
        double ReprojectErrorSelf_ = 0;

    };

    class Tracker{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Tracker> Ptr;

        enum class TrackerStatus
        {
            Ok, //tracking went smoothly
            Lost, //something went wrong
            NewKeyframe //tracking went smoothly but new keyframe required
        };

        Tracker(const SystemConfig::Ptr system_config,const CameraConfig::Ptr camera_config);
        ~Tracker();
        TrackerStatus TrackNewestFrame(Frame::Ptr& target_frame,const Frame::Ptr& host_frame,std::vector<SE3>& initialize_pose,SE3& pose_final_change);
        TrackerStatus TrackNewestFrameUsingLSD(Frame::Ptr& target_frame,const Frame::Ptr& host_frame,SE3& initialize_pose);
        TrackerStatus TrackNewestFrameUsingLSDOnLvl(Frame::Ptr& target_frame,const Frame::Ptr& host_frame,SE3& initialize_pose,int lvl);
        double TrackNewestError(Frame::Ptr& target_frame,const Frame::Ptr& host_frame,SE3& initialize_pose,int lvl);
        void ComputeJacobianSSE(lsd_slam::LGS6& ls,int lvl);

        TrackerStatus CheckTrackStatus(const Frame::Ptr& target_frame,const Frame::Ptr& host_frame,const SE3& initialize_pose,TrackerStatus status);
        TrackerStatus CheckKeyFrame(const Frame::Ptr& target_frame,const Frame::Ptr& host_frame,const SE3& initialize_pose);

        SE3 CheckBestInitPose(Frame::Ptr& target_frame,const Frame::Ptr& host_frame,std::vector<SE3>& initialize_pose);

        void ReprojectError(const Frame::Ptr& target_frame,const Frame::Ptr& host_frame,const SE3& initialize_pose);
        double TrackAverageError(const Frame::Ptr& target_frame,const Frame::Ptr& host_frame,const SE3& initialize_pose);
        double TrackAverageErrorLvl(const Frame::Ptr& target_frame,const Frame::Ptr& host_frame,const SE3& initialize_pose,int lvl);
        Vec3 InterpolateDTdxdy(const Vec2 Puv,const Vec3* DTInfo,int lvl);
        std::vector<SE3> TheLastTryPoses(std::vector<SE3> tryPoses);
        void DebugError(double e);

        SystemConfig::Ptr TrackConifg_;
        CameraConfig::Ptr CameraConfig_;
        BufferInfo* mBufferInfo_ = new BufferInfo();
        DeBugInfo* mDebuger_ = new DeBugInfo();

        mutable float* buf_warped_depth;
        mutable float* buf_warped_x;
        mutable float* buf_warped_y;
        mutable float* buf_warped_dx;
        mutable float* buf_warped_dy;
        mutable float* buf_warped_residual;
        mutable float* buf_warped_weight;
        std::mutex PoseDataMutex_;
    };
}

#endif //EDGEVO_TRACK_H
