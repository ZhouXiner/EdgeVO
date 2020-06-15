//
// Created by zhouxin on 2020/6/1.
//
#include "CeresTrack.h"

namespace EdgeVO{
    TrackCostFunction::TrackCostFunction(Vec2 point,Vec2 gradient,double depth,EdgeVO::Frame::Ptr &frame, EdgeVO::CameraConfig::Ptr &camera, int lvl) {
        TargetFrame_ = frame;
        Camera_ = camera;
        fx_ = Camera_->Fx_[lvl];
        fy_ = Camera_->Fy_[lvl];
        cx_ = Camera_->Cx_[lvl];
        cy_ = Camera_->Cy_[lvl];
        lvl_ = lvl;
        Puv_host_ = point;
        gradient_ = gradient;
        depth_ = depth;
    }

    bool TrackCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Vec6 se3;
        Vec2 target_edge,error_Vec;
        Vec2 g = gradient_.normalized();

        double weight = 1.0;
        se3 << parameters[0][0],parameters[0][1],parameters[0][2],
                parameters[0][3],parameters[0][4],parameters[0][5];

        SE3 Tij = SE3::exp(se3);
        Vec3 Pxyz_host = Camera_->pixel2camera(Puv_host_,depth_,lvl_);
        Vec3 Pxyz_target = Camera_->camera2camera(Pxyz_host,Tij);
        Vec2 Puv_target = Camera_->camera2pixel(Pxyz_target,lvl_);

        int x = static_cast<int>(Puv_target[0]);
        int y = static_cast<int>(Puv_target[1]);
        if(!Utility::InBorder(x,y,Camera_->SizeW_[lvl_],Camera_->SizeH_[lvl_],Camera_->ImageBound_) || Pxyz_target[2] <= 0){
            if(jacobians) {
                if (jacobians[0]) {
                    jacobians[0][0] = 0;
                    jacobians[0][1] = 0;
                    jacobians[0][2] = 0;
                    jacobians[0][3] = 0;
                    jacobians[0][4] = 0;
                    jacobians[0][5] = 0;
                }
            }
            residuals[0] = 0;
        }else {
            target_edge = GetNearestEdge(x, y);
            //std::cout << "host: " << Puv_host_.transpose() << " target: " << target_edge.transpose() << std::endl;
            ///error的形式，是需要好好考虑的，直接采用两个点之间的欧氏距离还是多添加一些，需要测试
            ///先按照欧式距离测试
            //gradient_.normalized();
            error_Vec = Puv_target - target_edge;

            double dis_g = gradient_.norm();
            //std::cout << "error: " << residuals[0] << " origin: " << dis_e << " near: " << dis_g << std::endl;
            //residuals[0] = (Puv_target - target_edge).norm();
            residuals[0] = error_Vec.dot(g);
            //residuals[0] = error_Vec.dot(error_Vec);
            //std::cout << "error: " << residuals[0] << std::endl;
            //residuals[0] = pow (error_Vec.dot(error_Vec),0.5);

            /*
            double huber = 1.5;
            double old_e = residuals[0];;
            if(abs(residuals[0]) > huber){
                weight = huber / abs(residuals[0]);
                residuals[0] = weight * residuals[0];
            }else{
                weight = 1.0;
            }
             */

            double v = 2.2875, theta = 1.105;
            if(residuals[0] > 1.5){
                weight = (v + 1) / (v + pow(residuals[0] / theta, 2));
                residuals[0] = weight * residuals[0];
            }else{
                weight = 1.0;
            }

            //std::cout << "old: " << old_e  << "        huber: " << weight << "       t: " << weights << std::endl;
        }

        if(jacobians){
            Eigen::Matrix<double,1,2> jaco;
            double dis = (error_Vec).dot(error_Vec);
            double dis_g2 = gradient_.dot(gradient_);
            //jaco << (target_edge[0] - Puv_target[0]) * pow(dis,-1/2) * weight,(target_edge[1] - Puv_target[1]) * pow(dis,-1/2) * weight;
            //jaco << 2 * error_Vec[0] * weight,2 * error_Vec[1] * weight;
            //jaco << (2 * weight * g[0] * (error_Vec[0] * g[0] + error_Vec[1] * g[1])),
            //        (2 * weight * g[1] * (error_Vec[0] * g[0] + error_Vec[1] * g[1]));
            jaco << g[0],g[1];
            if(jacobians[0]){
                Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> jacobian_pose_0(jacobians[0]);
                Eigen::Matrix<double,2,6> jaco_0;
                double x_2 = Pxyz_target[0] * Pxyz_target[0];
                double y_2 = Pxyz_target[1] * Pxyz_target[1];
                double z_2 = Pxyz_target[2] * Pxyz_target[2];
                jaco_0 << fx_ / Pxyz_target[2], 0, -fx_ * Pxyz_target[0] / z_2,-fx_ * Pxyz_target[0] * Pxyz_target[1] / z_2,fx_ + fx_ * x_2 / z_2, -fx_ * Pxyz_target[1] / Pxyz_target[2],
                          0, fy_ / Pxyz_target[2], -fy_ * Pxyz_target[1] / z_2, -fy_ - fy_ * y_2 / z_2, fy_ * Pxyz_target[0] * Pxyz_target[1] / z_2, fy_ * Pxyz_target[0] / Pxyz_target[2];
                jacobian_pose_0 = weight * jaco * jaco_0;
                //std::cout << "Jacobian: " << jacobian_pose_0 << std::endl;
            }
        }
        return true;
    }

    Vec2 TrackCostFunction::GetNearestEdge(int x, int y) const {
        Vec2 output;
        while(true){
            const int last_x = TargetFrame_->LocationX_[lvl_].at<int>(y,x);
            const int last_y = TargetFrame_->LocationY_[lvl_].at<int>(y,x);

            if(int(last_x) == x && int(last_y) == y){
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
