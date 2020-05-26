//
// Created by zhouxin on 2020/5/26.
//

#ifndef EDGEVO_SYSTEM_H
#define EDGEVO_SYSTEM_H
#include <iomanip>

#include "EdgeVO/CommonInclude.h"
#include "EdgeVO/Camera.h"
#include "EdgeVO/Config.h"
#include "EdgeVO/Inputs.h"
#include "EdgeVO/Frame.h"
#include "EdgeVO/Viewer.h"
#include "EdgeVO/Track.h"
#include "EdgeVO/BackEnd.h"


/**1.Reading Frame and IMU
 * 2.Running(Track(vo) + LocalMap + Looper + GlobalMap)*/

namespace EdgeVO{
    class Tracker;
    class LocalMapper;

    class System{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<System> Ptr;
        System();
        System(std::string config_path);
        void Run();
        void Tracking(Frame::Ptr& newestFrame);
        void TrackingNewestFrame(Frame::Ptr& newestFrame);
        void RecordAllTrajectory();
        void RecordSingleTrajectory(Frame::Ptr & newestKF);
        bool CheckRelocalisationMode(Frame::Ptr& newestFrame);
        void DebugShowRGBImage(Frame::Ptr& frame);

        enum class SystemStatus /**Depend how the system works*/
        {Init,
            Tracking,
            Relocalisation};


        SystemStatus  mSystemStatus_ = SystemStatus::Init;
        std::mutex TrajectoryMutex_;
        std::ofstream TrackOut_;
        std::ofstream SingleTrackOut_;

        std::unique_ptr<FrameDataset> mFrameDataset_;
        std::unique_ptr<Tracker> mTracker_;
        BackEnd::Ptr mBackEnd_;
        Viewer::Ptr mViewer_;

        SystemConfig::Ptr SystemConfig_;
        CameraConfig::Ptr CameraConfig_;

        bool LostInLastFrame_ = false;
    };
}
#endif //EDGEVO_SYSTEM_H
