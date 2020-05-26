//
// Created by zhouxin on 2020/5/26.
//

#ifndef EDGEVO_CAMERA_H
#define EDGEVO_CAMERA_H
#include "EdgeVO/CommonInclude.h"
#include "EdgeVO/Config.h"
/**该类中存放了不同数据集中，关于相机的基本信息。此外，构建了多层金字塔*/

namespace EdgeVO{
    class CameraConfig{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<CameraConfig> Ptr;
        CameraConfig(const SystemConfig::Ptr camera_config);
        void BuildPyramid();


        Vec2 camera2pixel(const Vec3 &p_c,int lvl = 0);

        Vec3 pixel2camera(const Vec2 &p_p, double depth = 1,int lvl = 0);

        Vec3 camera2camera(const Vec3 &p_i,const SE3 & Tij);

        Vec2 pixel2pixel(const Vec2 &p_i,const double depth ,const SE3 Tij, int lvl);

        int PyramidLevel_;
        int ImageBound_;

        Vec5 Distort_; /**(k1,k2,k3,p1,p2)*/
        std::vector<Mat33> K_;
        std::vector<Mat33> KInverse_;
        std::vector<double> Fx_,Fy_,Cx_,Cy_,FxInverse_,FyInverse_;
        std::vector<double> Weight_;
        std::vector<int> BoundH_, BoundW_, SizeW_, SizeH_;

        SystemConfig::Ptr CameraConfigConfig_;

    };
}
#endif //EDGEVO_CAMERA_H
