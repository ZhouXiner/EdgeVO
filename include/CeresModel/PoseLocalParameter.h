//
// Created by zhouxin on 2020/4/20.
//

#ifndef EDGEVO_POSELOCALPARAMETER_H
#define EDGEVO_POSELOCALPARAMETER_H

#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "EdgeVO/CommonInclude.h"
namespace EdgeVO{
    class PoseLocalParameter : public ceres::LocalParameterization
    {
        virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
        virtual bool ComputeJacobian(const double *x, double *jacobian) const;
        virtual int GlobalSize() const { return 6; };
        virtual int LocalSize() const { return 6; };
    };
}


#endif //EDGEVO_POSELOCALPARAMETER_H
