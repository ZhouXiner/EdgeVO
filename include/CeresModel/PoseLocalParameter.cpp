//
// Created by zhouxin on 2020/4/20.
//

#include "PoseLocalParameter.h"

namespace EdgeVO{
    bool PoseLocalParameter::ComputeJacobian(const double *x, double *jacobian) const {
        ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
        return true;
    }

    bool PoseLocalParameter::Plus(const double *x, const double *delta, double *x_plus_delta) const {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

        SE3 T = SE3::exp(lie);
        SE3 delta_T = SE3::exp(delta_lie);
        Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

        for(int i = 0; i < 6; ++i) x_plus_delta[i] = x_plus_delta_lie(i, 0);
        return true;
    }
}