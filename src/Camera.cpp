//
// Created by zhouxin on 2020/3/25.
//
#include "EdgeVO/Camera.h"

namespace EdgeVO{
    CameraConfig::CameraConfig(const SystemConfig::Ptr camera_config) {
        CameraConfigConfig_ = camera_config;
        BuildPyramid();
    }

    void CameraConfig::BuildPyramid() {
        PyramidLevel_ = CameraConfigConfig_->PyramidLevel_;
        ImageBound_ = CameraConfigConfig_->ImageBound_;
        K_.resize(PyramidLevel_);
        KInverse_.resize(PyramidLevel_);
        FxInverse_.resize(PyramidLevel_);
        FyInverse_.resize(PyramidLevel_);
        Fx_.resize(PyramidLevel_);
        Fy_.resize(PyramidLevel_);
        Cx_.resize(PyramidLevel_);
        Cy_.resize(PyramidLevel_);
        Weight_.resize(PyramidLevel_);
        BoundH_.resize(PyramidLevel_);
        BoundW_.resize(PyramidLevel_);
        SizeH_.resize(PyramidLevel_);
        SizeW_.resize(PyramidLevel_);

        int height,width;
        width = CameraConfigConfig_->ImageWidth_;
        height = CameraConfigConfig_->ImageHeight_;

        float fx,fy,cx,cy,k1,k2,k3,p1,p2;
        fx = CameraConfigConfig_->fx;
        fy = CameraConfigConfig_->fy;
        cx = CameraConfigConfig_->cx;
        cy = CameraConfigConfig_->cy;
        k1 = CameraConfigConfig_->k1;
        k2 = CameraConfigConfig_->k2;
        k3 = CameraConfigConfig_->k3;

        Distort_ << k1,k2,k3,p1,p2;
        double factor = 1 / CameraConfigConfig_->PyramidFactor_;
        for(size_t lvl = 0; lvl < PyramidLevel_;lvl++){

            factor = factor * CameraConfigConfig_->PyramidFactor_;
            double factorinverse = 1.0f / factor;


            /**Image Size Info*/
            SizeW_[lvl] = static_cast<int>(width * factorinverse);
            SizeH_[lvl] = static_cast<int>(height * factorinverse);

            BoundW_[lvl] = SizeW_[lvl] - CameraConfigConfig_->ImageBound_;
            BoundH_[lvl] = SizeH_[lvl] - CameraConfigConfig_->ImageBound_;

            /**CameraConfig Parameters*/
            Fx_[lvl] = fx * factorinverse; Fy_[lvl] = fy * factorinverse;
            FxInverse_[lvl] = (1.0f / fx) * factor; FyInverse_[lvl] = (1.0f / fy) * factor;
            Cx_[lvl] = cx * factorinverse; Cy_[lvl] = cy * factorinverse;

            K_[lvl] << Fx_[lvl], 0, Cx_[lvl], 0, Fy_[lvl], Cy_[lvl], 0, 0, 1;
            KInverse_[lvl] << FxInverse_[lvl], 0, -cx / fx, 0, FyInverse_[lvl], -cy / fy, 0,0,1;

            Weight_[lvl] = factor;
        }

    }

    Vec2 CameraConfig::camera2pixel(const Vec3 &p_c,int lvl) {
        return Vec2(
                Fx_[lvl] * p_c(0, 0) / p_c(2, 0) + Cx_[lvl],
                Fy_[lvl] * p_c(1, 0) / p_c(2, 0) + Cy_[lvl]
        );
    }

    Vec3 CameraConfig::pixel2camera(const Vec2 &p_p, double depth,int lvl) {
        return Vec3(
                (p_p(0, 0) - Cx_[lvl]) * depth / Fx_[lvl],
                (p_p(1, 0) - Cy_[lvl]) * depth / Fy_[lvl],
                depth
        );
    }

    Vec3 CameraConfig::camera2camera(const Vec3 &p_i, const SE3 &Tij) {
        const Mat33 R = Tij.rotationMatrix();
        const Vec3 t = Tij.translation();
        Vec3 p_new = R * p_i + t;
        return p_new;
    }

    Vec2 CameraConfig::pixel2pixel(const Vec2 &p_i, const double depth, const SE3 Tij, int lvl) {
        Vec3 pxyz_host = pixel2camera(p_i,depth,lvl);
        Vec3 pxyz_target = camera2camera(pxyz_host,Tij);
        Vec2 puv_target = camera2pixel(pxyz_target,lvl);
        return puv_target;
    }

}
