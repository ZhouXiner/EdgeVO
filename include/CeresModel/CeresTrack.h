//
// Created by zhouxin on 2020/6/1.
//

#ifndef EDGEVO_CERESTRACK_H
#define EDGEVO_CERESTRACK_H
#include <ceres/ceres.h>
#include "EdgeVO/CommonInclude.h"
#include "EdgeVO/Camera.h"
#include "EdgeVO/Frame.h"
#include "../Utils/Utility.h"

namespace EDGEVO{
    class TrackCostFunction : public ceres::SizedCostFunction<1,6>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        TrackCostFunction(Vec2 point,double depth,EdgeVO::Frame::Ptr& frame,EdgeVO::CameraConfig::Ptr& camera,int lvl);
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
        void check(double **parameters);
        bool InBorder(Vec2 Puv) const;
        Vec2 GetNearestEdge(int x,int y) const;

        Vec2 Puv_host_;
        double depth_;
        Vec3* DTInfo_;
        EdgeVO::CameraConfig::Ptr Camera_;
        EdgeVO::Frame::Ptr TargetFrame_;
        double weight_,fx_,fy_,cx_,cy_;
        int width_,lvl_;
        double costs = 0;
    };
}
#endif //EDGEVO_CERESTRACK_H
