//
// Created by zhouxin on 2020/5/26.
//

#ifndef EDGEVO_INPUTS_H
#define EDGEVO_INPUTS_H
#include <list>
#include "EdgeVO/CommonInclude.h"
#include "EdgeVO/Camera.h"
#include "EdgeVO/Config.h"
#include "EdgeVO/Frame.h"

namespace EdgeVO{
    class FrameDataset{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<FrameDataset> Ptr;
        FrameDataset();
        FrameDataset(const SystemConfig::Ptr system_config,const CameraConfig::Ptr camera);

        void Run(); /**using thread to get Images*/
        bool IsActive();
        void ReadImages();
        Frame::Ptr GetNewestFrame();

        std::list<Frame::Ptr> FramesList_;
        int ImgCount_;


    private:
        std::mutex FrameQueueLock_; /**Ensuring the push and pop operation accuracy*/
        bool IsActive_ = false;
        SystemConfig::Ptr ImageConfig_;
        CameraConfig::Ptr CameraConfig_;
        std::ifstream FileLists_; /**Only Dataset can visit the File*/
    };

    class ImuDataset{

    };

}
#endif //EDGEVO_INPUTS_H
