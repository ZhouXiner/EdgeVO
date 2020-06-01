//
// Created by zhouxin on 2020/5/26.
//

#ifndef EDGEVO_FRAME_H
#define EDGEVO_FRAME_H
#pragma once
#include "../Utils/Utility.h"
#include "EdgeVO/CommonInclude.h"
#include "EdgeVO/Camera.h"
#include "EdgeVO/Config.h"
#include "EdgeVO/EdgePixel.h"

namespace EdgeVO{
    class Frame{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;
        Frame();
        Frame(const SystemConfig::Ptr system_config,const CameraConfig::Ptr camera);
        void Initialize(const cv::Mat& rgbImg,const cv::Mat& depthImg,const double rgbStamp,const double depthStamp);

        void GetPyramidImgs(); /**Get the Edge, DT Imgs in each Pyramid*/
        void GetPyramidDTInfo(); /**For each DT Image, get the (dt,gredient_x,gredient_y)*/
        void GetEdgePixels();
        void GetLocation();
        void GetEdge(cv::Mat& EdgeImg,const cv::Mat &GrayImg);
        void GetDT(cv::Mat& DTImg,const cv::Mat &EdgeImg);

        Vec2 GetNearestEdge(int x,int y,int lvl);

        void DeBugImg();


        void UpdatePoseCW(const SE3& pose);
        void UpdatePoseC1C2(const SE3& pose);
        void UpdateHostId(const int id);
        void UpdateRefKF(Frame::Ptr& kf);

        SE3 ReturnPoseCW();
        SE3 ReturnPoseWC();
        SE3 ReturnPoseC1C2();
        Frame::Ptr ReturnRefKF();
        int ReturnId();
        int ReturnHostId();
        int ReturnPixelSize();
        int ReturnGoodSize();
        int ReturnBadSize();
        int ReturnBadAndUncertainSize();
        int ReturnUncertainSize();

        /**Identity*/
        long long int Id_;
        int KFId_;


        /**Images*/
        cv::Mat RGBInputImg_, DepthInputImg_;
        double RGBTimeStamp_,DepthTimeStamp_;


        std::vector<cv::Mat> RGBImgs_;
        std::vector<cv::Mat> DepthImgs_;
        std::vector<cv::Mat> GrayImgs_;
        std::vector<cv::Mat> EdgeImgs_;
        std::vector<cv::Mat> DTImgs_;
        std::vector<cv::Mat> LocationX_;
        std::vector<cv::Mat> LocationY_;
        /**Pyramid and Pixel Info*/
        std::vector<Vec3*> PyramidDT_; /**vec3f(DT,gredient_x,gredient_y)*/

        std::vector<std::list<EdgePixel::Ptr>> EdgePixels_;  /**pyramid Pixels*/
        std::vector<std::vector<int>> EdgeNumCountVec_;
        int PatchNumEachLvl_[3];

        /**Poses*/
        SE3 Tcw_;
        SE3 Tth_;
        int HostId_;
        Frame::Ptr RefKF_;

    private:
        SystemConfig::Ptr FrameConfig_;
        CameraConfig::Ptr CameraConfig_;
        std::mutex ImgDataMetux_;

    };
}
#endif //EDGEVO_FRAME_H
