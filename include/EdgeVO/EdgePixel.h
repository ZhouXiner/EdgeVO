//
// Created by zhouxin on 2020/5/26.
//

#ifndef EDGEVO_EDGEPIXEL_H
#define EDGEVO_EDGEPIXEL_H
#include "EdgeVO/CommonInclude.h"

namespace EdgeVO{
    class Frame;
    class EdgePixel{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<EdgePixel> Ptr;
        enum class EdgeStatus
        {
            Good, /// alreadly reprojected
            Uncertain,  /// never reprojected successfully
            Fail, /// update failed
            Marged ///be marged
        };

        EdgePixel(double x,double y,int lvl,double depth,double dt,double weight = 1.0f,Vec3 color = Vec3(0,0,0));
        Vec2 ReturnPixelPosition();
        Vec3 ReturnHomPosition();

        double Hostx_,Hosty_; /**Position in pixel Image*/
        int Lvl_; /**Which level*/
        Vec3 Color_;
        double Depth_;
        double DT_;
        double DepthInverse_;
        double Weight_;

        double theta_;
        double variance_;

        EdgeStatus EdgeStatus_;
        int UsedNum_ = 0;
    };
}
#endif //EDGEVO_EDGEPIXEL_H
