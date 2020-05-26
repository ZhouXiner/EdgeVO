//
// Created by zhouxin on 2020/3/26.
//
#include "EdgeVO/Track.h"

namespace EdgeVO{
    Tracker::Tracker(const SystemConfig::Ptr system_config,const CameraConfig::Ptr camera_config) {
        TrackConifg_ = system_config;
        CameraConfig_ = camera_config;
        float memSize = CameraConfig_->SizeW_[0] * CameraConfig_->SizeH_[0];
        buf_warped_residual = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
        buf_warped_dx = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
        buf_warped_dy = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
        buf_warped_x = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
        buf_warped_y = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
        buf_warped_depth = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
        buf_warped_weight = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    }

    Tracker::~Tracker(){
        Eigen::internal::aligned_free((void*)buf_warped_residual);
        Eigen::internal::aligned_free((void*)buf_warped_dx);
        Eigen::internal::aligned_free((void*)buf_warped_dy);
        Eigen::internal::aligned_free((void*)buf_warped_x);
        Eigen::internal::aligned_free((void*)buf_warped_y);
        Eigen::internal::aligned_free((void*)buf_warped_depth);
        Eigen::internal::aligned_free((void*)buf_warped_weight);
    }
    /**1.Get Best initialization pose  2.Calculate error   3.optimize error*/
    Tracker::TrackerStatus Tracker::TrackNewestFrame(EdgeVO::Frame::Ptr& target_frame,
                                                     const EdgeVO::Frame::Ptr& host_frame,
                                                     std::vector<SE3> &initialize_pose,SE3& pose_final_change) {
        mDebuger_->FrameCount_++;
        auto t1=std::chrono::steady_clock::now();
        SE3 BestInitPose = CheckBestInitPose(target_frame,host_frame,initialize_pose);
        auto t2=std::chrono::steady_clock::now();

        SE3 TrackPose = BestInitPose;
        TrackerStatus trackStatus = TrackNewestFrameUsingLSD(target_frame,host_frame,TrackPose);

        auto t3=std::chrono::steady_clock::now();

        pose_final_change = TrackPose;

        TrackerStatus TestStatus = CheckTrackStatus(target_frame,host_frame,TrackPose,trackStatus);

        ///如果失败了，我们最后给它一次机会，在２７个方向上重新track
        if(TestStatus == TrackerStatus::Lost){
            std::vector<SE3> LastTryPoses = TheLastTryPoses(initialize_pose);
            for(auto& pose:LastTryPoses){
                TrackerStatus tmpstatus = TrackNewestFrameUsingLSD(target_frame,host_frame,pose);
                tmpstatus = CheckTrackStatus(target_frame,host_frame,pose,tmpstatus);
                if(tmpstatus != TrackerStatus::Lost){
                    pose_final_change = TrackPose;
                    return tmpstatus;
                }
            }
        }else{
            return TestStatus;
        }
        return CheckTrackStatus(target_frame,host_frame,TrackPose,trackStatus);
    }

    SE3 Tracker::CheckBestInitPose(Frame::Ptr& target_frame, const Frame::Ptr& host_frame,
                                   std::vector<SE3>& initialize_pose) {
        double min_cost = 9999,min_id;
        SE3 pose;
        for(size_t i = 0;i<initialize_pose.size();i++){
            double cost = TrackAverageError(target_frame,host_frame,initialize_pose[i]);
            //std::cout << "id: " << target_frame->Id_ << " cost: " << cost << " which: " << i << std::endl;
            if(min_cost > cost){
                min_cost = cost;
                min_id = i;

            }
        }
        pose = initialize_pose[min_id];
        LOG(INFO) << "Check the best initialization " << min_id;
        return pose;
    }

    double Tracker::TrackAverageErrorLvl(const EdgeVO::Frame::Ptr &target_frame, const EdgeVO::Frame::Ptr &host_frame,
                                         const SE3 &initialize_pose, int lvl) {
        double error = 0;
        int goodEdge = 0;
        for(auto &pixel : host_frame->EdgePixels_[lvl]){
            Vec2 Puv_host = pixel->ReturnPixelPosition();
            double depth_host = pixel->Depth_;
            Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host,depth_host,lvl);

            Mat33 R_t_h = initialize_pose.rotationMatrix();
            Vec3 T_t_h = initialize_pose.translation();
            Vec3 Pxyz_target = R_t_h * Pxyz_host + T_t_h;
            Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target,lvl);


            if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0){
                Vec3 dtInfo = InterpolateDTdxdy(Puv_target,target_frame->PyramidDT_[lvl],lvl);

                if(std::isnan(dtInfo[0]) || std::isnan(dtInfo[1]) || std::isnan(dtInfo[2])){
                    LOG(ERROR) << "Nan Data!";
                    exit(0);
                }

                //double e = dtInfo[0] * CameraConfig_->Weight_[lvl];

                double e = dtInfo[0];
                if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                    continue;
                }

                if(std::isnan(e)){
                    LOG(ERROR) << "Bad info " << Puv_target << CameraConfig_->SizeW_[lvl] << " " << CameraConfig_->SizeH_[lvl];
                    exit(0);
                }
                error = error + e;
                ++goodEdge;
            }
        }
        if(goodEdge == 0){
            LOG(ERROR) << "Test no good edge !" << " id: " << target_frame->Id_ << std::endl;
            return MAXFLOAT;
        }
        return error / static_cast<double>(goodEdge);
    }

    double Tracker::TrackAverageErrorForLoop(const EdgeVO::Frame::Ptr &target_frame,
                                             const EdgeVO::Frame::Ptr &host_frame, const SE3 &initialize_pose,
                                             EdgeVO::BufferInfo &residual_info) {
        residual_info.Reset();
        for(size_t lvl = 0;lvl < TrackConifg_->PyramidLevel_;++lvl){
            for(auto &pixel : host_frame->EdgePixels_[lvl]){
                Vec2 Puv_host = pixel->ReturnPixelPosition();
                double depth_host = pixel->Depth_;
                Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host,depth_host,lvl);

                Mat33 R_t_h = initialize_pose.rotationMatrix();
                Vec3 T_t_h = initialize_pose.translation();
                Vec3 Pxyz_target = R_t_h * Pxyz_host + T_t_h;
                Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target,lvl);


                if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0){
                    Vec3 dtInfo = InterpolateDTdxdy(Puv_target,target_frame->PyramidDT_[lvl],lvl);

                    if(std::isnan(dtInfo[0]) || std::isnan(dtInfo[1]) || std::isnan(dtInfo[2])){
                        LOG(ERROR) << "Nan Data!";
                        exit(0);
                    }

                    //double e = dtInfo[0] * CameraConfig_->Weight_[lvl];

                    double e = dtInfo[0];
                    if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                        residual_info.BadEdgesNum_++;
                        continue;
                    }

                    if(std::isnan(e)){
                        LOG(ERROR) << "Bad info " << Puv_target << CameraConfig_->SizeW_[lvl] << " " << CameraConfig_->SizeH_[lvl];
                        exit(0);
                    }
                    residual_info.SumError_ = residual_info.SumError_ + e;
                    residual_info.GoodEdgesNum_++;
                }
            }
        }
        if(residual_info.GoodEdgesNum_ == 0){
            LOG(ERROR) << "Init no good edge !" << " host id: " << host_frame->Id_ << " target id: " << target_frame->Id_ << std::endl;
            return MAXFLOAT;
        }

        //LOG(INFO) << "Good Size: " << residual_info.GoodEdgesNum_ << std::endl;
        return residual_info.SumError_ / static_cast<double>(residual_info.GoodEdgesNum_);
    }
    double Tracker::TrackAverageError(const EdgeVO::Frame::Ptr& target_frame, const EdgeVO::Frame::Ptr& host_frame,
                                      const SE3& initialize_pose) {
        double error = 0;
        int goodEdge = 0;
        for(size_t lvl = 0;lvl < TrackConifg_->PyramidLevel_;++lvl){
            for(auto &pixel : host_frame->EdgePixels_[lvl]){
                Vec2 Puv_host = pixel->ReturnPixelPosition();
                double depth_host = pixel->Depth_;
                Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host,depth_host,lvl);

                Mat33 R_t_h = initialize_pose.rotationMatrix();
                Vec3 T_t_h = initialize_pose.translation();
                Vec3 Pxyz_target = R_t_h * Pxyz_host + T_t_h;
                Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target,lvl);


                if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0){
                    Vec3 dtInfo = InterpolateDTdxdy(Puv_target,target_frame->PyramidDT_[lvl],lvl);

                    if(std::isnan(dtInfo[0]) || std::isnan(dtInfo[1]) || std::isnan(dtInfo[2])){
                        LOG(ERROR) << "Nan Data!";
                        exit(0);
                    }

                    //double e = dtInfo[0] * CameraConfig_->Weight_[lvl];

                    double e = dtInfo[0];
                    if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                        continue;
                    }

                    if(std::isnan(e)){
                        LOG(ERROR) << "Bad info " << Puv_target << CameraConfig_->SizeW_[lvl] << " " << CameraConfig_->SizeH_[lvl];
                        exit(0);
                    }
                    error = error + e;
                    goodEdge++;
                }
            }
        }
        if(goodEdge == 0){
            LOG(ERROR) << "Init no good edge !" << " host id: " << host_frame->Id_ << " target id: " << target_frame->Id_ << std::endl;
            return MAXFLOAT;
        }
        return error / static_cast<double>(goodEdge);
    }

    void Tracker::ReprojectError(const EdgeVO::Frame::Ptr& target_frame, const EdgeVO::Frame::Ptr& host_frame,
                                 const SE3 &initialize_pose) {

        int lvl = 0;
        //mBufferInfo_->Reset();
        for(auto &pixel : host_frame->EdgePixels_[lvl]){
            Vec2 Puv_host = pixel->ReturnPixelPosition();
            double depth_host = pixel->Depth_;
            Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host,depth_host,lvl);

            Mat33 R_t_h = initialize_pose.rotationMatrix();
            Vec3 T_t_h = initialize_pose.translation();
            Vec3 Pxyz_target = R_t_h * Pxyz_host + T_t_h;
            Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target,lvl);

            if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0){
                Vec3 dtInfo = InterpolateDTdxdy(Puv_target,target_frame->PyramidDT_[lvl],lvl);

                if(std::isnan(dtInfo[0]) || std::isnan(dtInfo[1]) || std::isnan(dtInfo[2])){
                    LOG(ERROR) << "Nan Data!";
                    exit(0);
                }

                double e = dtInfo[0];

                if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                    continue;
                }

                if(std::isnan(e)){
                    LOG(ERROR) << "Bad info " << Puv_target << CameraConfig_->SizeW_[lvl] << " " << CameraConfig_->SizeH_[lvl];
                    exit(0);
                }
                //++mBufferInfo_->GoodEdgesNum_;
                mBufferInfo_->ReprojectError_ += (Puv_host - Puv_target).norm();
                mBufferInfo_->DTError_ += e;
                Mat33 R_identity = Mat33::Identity();
                Vec3 Pxyz_translation = R_identity * Pxyz_host + T_t_h;
                Vec2 Puv_host_reproject = CameraConfig_->camera2pixel(Pxyz_translation,lvl);

                mBufferInfo_->ReprojectErrorSelf_ += (Puv_host - Puv_host_reproject).norm();
            }
        }
        return;
    }
    Vec3 Tracker::InterpolateDTdxdy(const Vec2 Puv, const Vec3 * DTInfo,int lvl) {
        const int u_int = static_cast<int>(Puv(0));
        const int v_int = static_cast<int>(Puv(1));

        const double du = Puv[0] - u_int;
        const double dv = Puv[1] - v_int;
        const int width = CameraConfig_->SizeW_[lvl];
        const Vec3* baseVec = DTInfo + u_int + v_int*width;

        Vec3 result = du*dv * *(const Vec3*)(baseVec + 1 + width) + (dv - du*dv) * *(const Vec3*)(baseVec + width) +
                      (du - du*dv) * *(const Vec3*)(baseVec + 1) + (1-du-dv + du*dv) * *(const Vec3*)baseVec;

        return result;
    }

    Tracker::TrackerStatus Tracker::TrackNewestFrameUsingLSD(EdgeVO::Frame::Ptr& target_frame,
                                                             const EdgeVO::Frame::Ptr& host_frame,
                                                             SE3 &initialize_pose) {

        SE3 back_pose = initialize_pose;
        TrackerStatus trackStatus_final = TrackerStatus::Lost;
        TrackerStatus trackStatus;

        for(int lvl = TrackConifg_->PyramidLevel_ - 1;lvl >= 0;--lvl){
            if(host_frame->EdgePixels_[lvl].size() < TrackConifg_->TrackLeastNum_[lvl]){
                LOG(WARNING) << "Too Few Edges in Frame: " << target_frame->Id_  << " Lvl: " << lvl << " Num: " << host_frame->EdgePixels_[lvl].size();
                continue;
            }

            back_pose = initialize_pose;

            trackStatus = TrackNewestFrameUsingLSDOnLvl(target_frame,host_frame,initialize_pose,lvl);

            if(trackStatus != TrackerStatus::Ok){
                initialize_pose = back_pose;
                //LOG(WARNING) << "No improvement in ("<< target_frame->Id_  << "," << lvl << ")";
            } else{
                trackStatus_final = trackStatus;
            }
        }

        if(trackStatus_final == TrackerStatus::Lost){
            return TrackerStatus::Lost;
        }

        double e = TrackAverageError(target_frame,host_frame,initialize_pose);
        LOG(INFO) << "Feature Size: " << host_frame->ReturnPixelSize() << " After Track Lost: " << e << " id: " <<  target_frame->Id_ << " tracked in " << host_frame->Id_ << " with Good: " << mBufferInfo_->GoodEdgesNum_ ;

        //if(e > TrackConifg_->TrackMaxError_){
        //    LOG(INFO) << "bading tracking final error: " << e;
        //    return TrackerStatus::Lost;
        //}

        return trackStatus_final;
    }

    Tracker::TrackerStatus Tracker::TrackNewestFrameUsingLSDOnLvl(EdgeVO::Frame::Ptr& target_frame,
                                                                  const EdgeVO::Frame::Ptr& host_frame,
                                                                  SE3 &initialize_pose, int lvl) {
        lsd_slam::LGS6 ls;
        if(host_frame->ReturnPixelSize() < 200){
            LOG(ERROR) << "LOST because too few points" << std::endl;
        }

        /**optimzation*/
        //std::cout << "ok" << std::endl;
        auto lastError = TrackNewestError(target_frame,host_frame,initialize_pose,lvl);
        //std::cout << "ok1" << std::endl;

        auto error = lastError;
        auto lambda = 0.0f;
        TrackerStatus status = TrackerStatus ::Lost;
        for(size_t iter = 0;iter < TrackConifg_->TrackIters_;++iter){
            //std::cout << "ok here" << std::endl;

            if(mBufferInfo_->GoodEdgesNum_ < 100){
                LOG(ERROR) << "Lost because too few good points: " << mBufferInfo_->GoodEdgesNum_ << std::endl;
                return TrackerStatus::Lost;
            }

            ComputeJacobianSSE(ls,lvl);
            int incTry = 0;

            while(true){
                const Vec6 b = ls.b.cast<double>();
                Mat66 A = ls.A.cast<double>();
                for(size_t i = 0; i < 6; i++) A(i,i) *= 1+lambda;
                const Vec6 inc = A.ldlt().solve(-b);

                incTry++;
                //Make sure that Eigen doesn't crash the program
                if (!inc.allFinite()) return TrackerStatus::Lost;
                SE3 T_t_h_new = SE3::exp(inc) * initialize_pose;
                error = TrackNewestError(target_frame,host_frame,T_t_h_new,lvl);
                //std::cout << "inc: " << inc.transpose() << " error: " << error << " lastError: " << lastError << std::endl;
                if(error < lastError){
                    initialize_pose = T_t_h_new;
                    //std::cout  << "Increment accepted at iteration : "<<iter << " lambda: " << lambda << " error: " << error << " last error: " << lastError  << std::endl;
                    status = TrackerStatus ::Ok;
                    if(error / lastError > 0.999f){
                        iter = TrackConifg_->TrackIters_;
                    }
                    lastError = error;
                    lambda = (lambda <= 0.2f ? 0.0f : lambda*0.5f);
                } else{
                    if(!(inc.dot(inc) > 1e-16)){
                        iter = TrackConifg_->TrackIters_;
                        //std::cout << "StepSize too small" << std::endl;
                        break;
                    }
                    lambda = (lambda < 0.001f ? 0.2f : lambda * (1 << incTry));
                    //std::cout << "Increment NOT accepted at iteration : "<< iter << " lambda: " << lambda << std::endl;
                }
            }
        }

        return status;
    }

    void Tracker::ComputeJacobianSSE(lsd_slam::LGS6 &ls, int lvl) {
        const __m128 fx = _mm_set1_ps(CameraConfig_->Fx_[lvl]);
        const __m128 fy = _mm_set1_ps(CameraConfig_->Fy_[lvl]);


        ls.initialize();
        size_t idx = 0;
/*
        for(; idx < mBufferInfo_->GoodEdgesNum_; idx+=4)
        {
            const __m128 x = _mm_load_ps(buf_warped_x+idx);
            const __m128 y = _mm_load_ps(buf_warped_y+idx);
            // redefine pz
            const __m128 depth = _mm_load_ps(buf_warped_depth+idx);
            const __m128 dx = _mm_load_ps(buf_warped_dx+idx); //TODO: remove foxal length multi
            const __m128 fxdx = _mm_mul_ps(fx,dx);
            const __m128 J60 = _mm_div_ps(fxdx,depth); //id * dx * fx

            const __m128 dy = _mm_load_ps(buf_warped_dy+idx);
            const __m128 fydy = _mm_mul_ps(fy,dy);
            const __m128 J61 = _mm_div_ps(fydy,depth); //iD*gy *fy

            const __m128 fxdxx = _mm_mul_ps(fxdx,x); //fx*dx*x
            const __m128 fydyy = _mm_mul_ps(fydy,y); //fy*dy*y
            const __m128 dd = _mm_mul_ps(depth,depth);

            //Note: _mm_xor_ps(vec, _mm_set1_ps(-0.f)) flips the singn
            const __m128 J62 = _mm_div_ps(_mm_sub_ps(_mm_xor_ps(fxdxx, _mm_set1_ps(-0.f)),fydyy),dd);

            const __m128 fydy_div_1_and_yydd = _mm_sub_ps(_mm_xor_ps(fydy, _mm_set1_ps(-0.f)),_mm_div_ps(_mm_mul_ps(fydyy,y),dd));
            const __m128 J63 = _mm_sub_ps(fydy_div_1_and_yydd,_mm_div_ps(_mm_mul_ps(fxdxx,y),dd));

            const __m128 fxdx_div_1_and_xxdd = _mm_add_ps(fxdx,_mm_div_ps(_mm_mul_ps(fxdxx,x),dd));
            const __m128 J64 = _mm_add_ps(fxdx_div_1_and_xxdd,_mm_div_ps(_mm_mul_ps(fydyy,x),dd));

            const __m128 J65 = _mm_sub_ps( _mm_div_ps(_mm_mul_ps(fydy,x),depth), _mm_div_ps(_mm_mul_ps(fxdx,y),depth));
            ls.updateSSE(J60, J61, J62, J63, J64, J65, _mm_load_ps(buf_warped_residual +idx), _mm_load_ps(buf_warped_weight+idx));
        }
        */
        // solve ls
        const float fx_f = CameraConfig_->Fx_[lvl];
        const float fy_f = CameraConfig_->Fy_[lvl];

        for(;idx < mBufferInfo_->GoodEdgesNum_;++idx){

            const float dx = buf_warped_dx[idx];
            const float dy = buf_warped_dy[idx];
            const float x =  buf_warped_x[idx];
            const float y = buf_warped_y[idx];
            const float z = buf_warped_depth[idx];

            lsd_slam::Vector6 J;
            J[0] = fx_f * dx / z;
            J[1] = fy_f * dy / z;
            J[2] = (-1 / (z * z)) * (fx_f * x * dx + fy_f * y * dy);
            J[3] = -fx_f * dx * x * y / (z*z)  - fy_f * dy * (1 + (y*y) / (z*z));
            J[4] = fx_f * dx * (1 + (x*x) / (z*z)) + fy_f * dy * (x*y)/(z*z);
            J[5] = -dx * fx_f * y / z + fy_f * x * dy / z;
            ls.update(J,buf_warped_residual[idx],buf_warped_weight[idx]);
        }
        ls.finish();
    }


    double Tracker::TrackNewestError(EdgeVO::Frame::Ptr &target_frame, const EdgeVO::Frame::Ptr &host_frame,
                                     SE3 &initialize_pose, int lvl) {

        Vec3* DTInfo_ = target_frame->PyramidDT_[lvl];

        mBufferInfo_->Reset();

        std::vector<double> ErrorVec;
        for(auto &pixel : host_frame->EdgePixels_[lvl]){
            if(pixel->Depth_ < 0) continue;
            Vec2 Puv_host = pixel->ReturnPixelPosition();
            double depth_host = pixel->Depth_;
            Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host,depth_host,lvl);
            Vec3 Pxyz_target = CameraConfig_->camera2camera(Pxyz_host,initialize_pose);
            Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target,lvl);

            if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0) {

                const auto dtInfo = InterpolateDTdxdy(Puv_target,DTInfo_,lvl);

                double e = dtInfo[0];
                float huber_weight = Utility::GetHuberWeight(e,TrackConifg_->TrackHuberWeright_);

                if(std::isnan(dtInfo[0]) || std::isnan(dtInfo[1]) || std::isnan(dtInfo[2])){
                    LOG(ERROR) << "Nan Data!";
                    exit(0);
                }

                if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                    ++mBufferInfo_->BadEdgesNum_;
                    continue;
                }
                buf_warped_x[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(Pxyz_target[0]);
                buf_warped_y[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(Pxyz_target[1]);
                buf_warped_depth[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(Pxyz_target[2]);
                buf_warped_residual[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(e);
                buf_warped_dx[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(dtInfo[1]);
                buf_warped_dy[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(dtInfo[2]);
                buf_warped_weight[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(huber_weight);
                mBufferInfo_->SumError_ = mBufferInfo_->SumError_ + e;
                ErrorVec.push_back(e);
                ++mBufferInfo_->GoodEdgesNum_;
            } else{
                ++mBufferInfo_->BadEdgesNum_;
                continue;
            }

        }

        //if(mBufferInfo_->GoodEdgesNum_ < 200) std::cout << "little num " << mBufferInfo_->GoodEdgesNum_ << " " << target_frame->Id_ << std::endl;
        if(mBufferInfo_->GoodEdgesNum_){
            std::sort(ErrorVec.begin(),ErrorVec.end());
            double mid_e = ErrorVec[static_cast<int>(mBufferInfo_->GoodEdgesNum_ / 2)];
            std::vector<double> MADVec;
            for(size_t i = 0;i<mBufferInfo_->GoodEdgesNum_;i++){
                if(i < mBufferInfo_->GoodEdgesNum_ / 2)
                    MADVec.push_back(mid_e - ErrorVec[i]);
                else
                    MADVec.push_back(ErrorVec[i] - mid_e);
            }
            std::sort(MADVec.begin(),MADVec.end());
            double MAD_theta = MADVec[static_cast<int>(mBufferInfo_->GoodEdgesNum_ / 2)];
            //std::cout << "MAD theta: " << MAD_theta << std::endl;
            for(size_t i =0;i<mBufferInfo_->GoodEdgesNum_;++i){
                buf_warped_weight[i] = Utility::GetTDistributionWeight(buf_warped_residual[i],1.4826 * MAD_theta);
            }
        }

        if(mBufferInfo_->GoodEdgesNum_ != 0){
            //std::cout << "Good num: " << mBufferInfo_->GoodEdgesNum_ << std::endl;
            return mBufferInfo_->SumError_ / static_cast<float>(mBufferInfo_->GoodEdgesNum_);
        }else{
            //std::cout << "Fuck num: " << mBufferInfo_->GoodEdgesNum_ << std::endl;
            return 99999;
        }
    }


    std::vector<SE3> Tracker::TheLastTryPoses(std::vector<SE3> tryPoses) {
        std::vector<SE3> lastF_2_fh_tries;

        for(auto& tryPose:tryPoses){
            SE3 try_init = tryPose;
            for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
            {
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                lastF_2_fh_tries.push_back(try_init * SE3(Eigen::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            }
        }
        return lastF_2_fh_tries;
    }

    Tracker::TrackerStatus Tracker::CheckTrackStatus(const EdgeVO::Frame::Ptr& target_frame, const EdgeVO::Frame::Ptr& host_frame,

                                                     const SE3 &initialize_pose, EdgeVO::Tracker::TrackerStatus status) {

        return CheckKeyFrame(target_frame,host_frame,initialize_pose);
        /*
        switch(status){
            case TrackerStatus ::Lost:
            {
                ///lost because no update
                double final_e = TrackAverageError(target_frame,host_frame,initialize_pose);
                if(final_e < TrackConifg_->TrackMaxError_){
                    return TrackerStatus::Ok;
                }
                LOG(INFO) << "Lost for no update and Lost too big! Id: " << target_frame->Id_ << std::endl;
                mDebuger_->LostNumAfterCeres_++;
                return TrackerStatus ::Lost;
            }
            case TrackerStatus ::Ok:{
                return CheckKeyFrame(target_frame,host_frame,initialize_pose);
            }
        }
          */
}

    Tracker::TrackerStatus Tracker::CheckKeyFrame(const EdgeVO::Frame::Ptr& target_frame,const EdgeVO::Frame::Ptr& host_frame,
                                                  const SE3 &initialize_pose) {
        double final_e = TrackAverageError(target_frame,host_frame,initialize_pose);
        mDebuger_->AllAfterError_ += final_e;
        DebugError(final_e);
        if(final_e > TrackConifg_->TrackMaxError_){
            LOG(INFO) << "Lost for big error! Id: " << target_frame->Id_ << " e: " <<  final_e;
            mDebuger_->LostNumAfterCheck_++;
            return TrackerStatus ::Lost;
        }
        /**
         * 1.视差
         * 2.图像的相似程度(直方图、特征点)
         * 3.光流移动的大小
        */
        /**optical error*/
        ReprojectError(target_frame,host_frame,initialize_pose);
        double average_parallax_error = mBufferInfo_->ReprojectError_ / mBufferInfo_->GoodEdgesNum_;
        double average_self_parallax_error = mBufferInfo_->ReprojectErrorSelf_ / mBufferInfo_->GoodEdgesNum_;

        ///丰富纹理下，不靠谱，但是在弱纹理环境下比较靠谱
        double average_dt_error = mBufferInfo_->DTError_ / mBufferInfo_->GoodEdgesNum_;
        auto opticalThreshold = std::sqrt(average_parallax_error)*0.1 + std::sqrt(average_self_parallax_error)*0.15;
        //std::cout << "optical: " << mBufferInfo_.ReprojectError_ << " " << mBufferInfo_.GoodEdgesNum_ << " " << average_parallax_error << std::endl;
        /**帧之间的差距*/
        int frame_count = target_frame->Id_ - host_frame->Id_;

        /**相似性*/

        //|| average_dt_error > 2.0 || average_self_parallax_error > 10
        if(average_parallax_error > TrackConifg_->TrackParallaxError_){
            mDebuger_->KeyNum_++;
            //cv::imshow("before",host_frame->RGBImgs_[0]);
            //cv::imshow("after",target_frame->RGBImgs_[0]);
            //cv::waitKey(100);
            //std::cout << "KeyFrame id: " << target_frame->Id_ << " average_dt_error: " << average_dt_error << std::endl;
            //std::cout << "KeyFrame id: " << target_frame->Id_ << " self_parallax_error: " << average_self_parallax_error << std::endl;
            //LOG(INFO) << "KeyFrame id: " << target_frame->Id_ << " average error: " << average_parallax_error;
            //std::cout << "optical error: " << opticalThreshold << std::endl;
            return TrackerStatus::NewKeyframe;
        }

        if(mBufferInfo_->GoodEdgesNum_ < mBufferInfo_->BadEdgesNum_ * 2) {
            //LOG(INFO) << "KeyFrame for few good: " << "(g,b)  (" << mBufferInfo_->GoodEdgesNum_ << "," << mBufferInfo_->BadEdgesNum_ << ")";
            return TrackerStatus::NewKeyframe;
        }
        ++mDebuger_->GoodNum_;
        //LOG(INFO) << "Just Ok! Id: " << target_frame->Id_;
        return TrackerStatus ::Ok;
    }

    void Tracker::DebugError(double e) {
        if(e < mDebuger_->MinError_){
            mDebuger_->MinError_ = e;
        }
        if(e > mDebuger_->MaxError_){
            mDebuger_->MaxError_ = e;
        }
    }
}


