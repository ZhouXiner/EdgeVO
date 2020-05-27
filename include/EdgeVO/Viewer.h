//
// Created by zhouxin on 2020/5/26.
//

#ifndef EDGEVO_VIEWER_H
#define EDGEVO_VIEWER_H
#include <cmath>
#include <pangolin/pangolin.h>

#include "EdgeVO/CommonInclude.h"
#include "EdgeVO/Frame.h"


namespace EdgeVO{
    class Viewer{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Viewer> Ptr;

        Viewer();

        void AddCurrentFrame(const Frame::Ptr& current_frame);
        void AddCurrentPose();

        void Update();
        void Close();
    private:
        void ThreadLoop();
        void DrawTrajectory();

        void DrawKeyFrames();
        void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);
        cv::Mat PlotFrameImage();

        std::vector<Vec3, Eigen::aligned_allocator<Vec3>> Translations_;

        std::vector<Frame::Ptr> ActiveKeyFrames_;
        Frame::Ptr CurrentFrame_ = nullptr;
        std::thread ViewerThread_;

        bool ViewerRunning_ = true;

        std::mutex ViewerDataMutex_;
    };
}
#endif //EDGEVO_VIEWER_H
