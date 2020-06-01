//
// Created by zhouxin on 2020/6/1.
//
#include "CeresTrack.h"

namespace EDGEVO{
    TrackCostFunction::TrackCostFunction(Vec2 point,double depth,EdgeVO::Frame::Ptr &frame, EdgeVO::CameraConfig::Ptr &camera, int lvl) {
        TargetFrame_ = frame;
        Camera_ = camera;
        fx_ = Camera_->Fx_[lvl];
        fy_ = Camera_->Fy_[lvl];
        cx_ = Camera_->Cx_[lvl];
        cy_ = Camera_->Cy_[lvl];
        lvl_ = lvl;
        Puv_host_ = point;
        depth_ = depth;
    }

    bool TrackCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Vec6 se3;
        Vec2 target_edge;
        se3 << parameters[0][0],parameters[0][1],parameters[0][2],
                parameters[0][3],parameters[0][4],parameters[0][5];

        SE3 Tij = SE3::exp(se3);
        Vec3 Pxyz_host = Camera_->pixel2camera(Puv_host_,depth_,lvl_);
        Vec3 Pxyz_target = Camera_->camera2camera(Pxyz_host,Tij);
        Vec2 Puv_target = Camera_->camera2pixel(Pxyz_target,lvl_);

        int x = static_cast<int>(Puv_target[0]);
        int y = static_cast<int>(Puv_target[1]);
        if(!Utility::InBorder(x,y,Camera_->SizeW_[lvl_],Camera_->SizeH_[lvl_],Camera_->ImageBound_)){
            if(jacobians) {
                if (jacobians[0]) {
                    jacobians[0][0] = 0;
                    jacobians[0][1] = 0;
                    jacobians[0][2] = 0;
                    jacobians[0][3] = 0;
                    jacobians[0][4] = 0;
                    jacobians[0][5] = 0;
                }
                residuals[0] = 0;
            }
        }else{
            Vec2 target_edge = GetNearestEdge(x,y);
            ///error的形式，是需要好好考虑的，直接采用两个点之间的欧氏距离还是多添加一些，需要测试
            ///先按照欧式距离测试
            residuals[0] = (Puv_host_ - target_edge).norm();
        }
        if(jacobians){
            Eigen::Matrix<double,1,2> jaco;
            double dis = (Puv_host_ - target_edge).dot(Puv_host_ - target_edge);
            jaco << (target_edge[0] - Puv_host_[0]) * pow(dis,-1/2),(target_edge[1] - Puv_host_[1]) * pow(dis,-1/2);
            if(jacobians[0]){
                Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> jacobian_pose_0(jacobians[0]);
                Eigen::Matrix<double,2,6> jaco_0;
                double x_2 = Pxyz_target[0] * Pxyz_target[0];
                double y_2 = Pxyz_target[1] * Pxyz_target[1];
                double z_2 = Pxyz_target[2] * Pxyz_target[2];
                jaco_0 << fx_ / Pxyz_target[2], 0, -fx_ * Pxyz_target[0] / z_2,-fx_ * Pxyz_target[0] * Pxyz_target[1] / z_2,fx_ + fx_ * x_2 / z_2, -fx_ * Pxyz_target[1] / Pxyz_target[2],
                          0, fy_ / Pxyz_target[2], -fy_ * Pxyz_target[1] / z_2, -fy_ - fy_ * y_2 / z_2, fy_ * Pxyz_target[0] * Pxyz_target[1] / z_2, fy_ * Pxyz_target[0] / Pxyz_target[2];
                jacobian_pose_0 = jaco * jaco_0;
            }
        }
        return true;
    }

    Vec2 TrackCostFunction::GetNearestEdge(int x, int y) const {
        Vec2 output;
        while(true){
            const uint8_t last_x = TargetFrame_->LocationX_[lvl_].at<uint8_t>(y,x);
            const uint8_t last_y = TargetFrame_->LocationX_[lvl_].at<uint8_t>(y,x);
            if(last_x == x && last_y == y){
                output[0] = x;
                output[1] = y;
                return output;
            }else{
                x = last_x;
                y = last_y;
            }
        }
    }
}
