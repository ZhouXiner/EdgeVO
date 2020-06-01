//
// Created by zhouxin on 2020/3/26.
//
#include "EdgeVO/EdgePixel.h"

namespace EdgeVO{
    EdgePixel::EdgePixel(double x, double y, int lvl, double depth,double dt, double weight, Vec3 color) {
        Hostx_ = x;
        Hosty_ = y;
        Lvl_ = lvl;
        Depth_ = depth;
        DepthInverse_ = 1.0f / depth;
        DT_ = dt;
        Weight_ = weight;
        Color_ = color;
        EdgeStatus_ = EdgeStatus::Uncertain;
    }

    Vec2 EdgePixel::ReturnPixelPosition() {
        return Vec2(Hostx_,Hosty_);
    }

    Vec3 EdgePixel::ReturnHomPosition() {
        return Vec3(Hostx_,Hosty_,1);
    }
}
