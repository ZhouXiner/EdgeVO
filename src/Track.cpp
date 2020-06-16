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

        /*
        TrackerStatus TestStatus = CheckTrackStatus(target_frame,host_frame,TrackPose,trackStatus);

        ///如果失败了，我们最后给它一次机会，在２７个方向上重新track

        if(TestStatus == TrackerStatus::Lost){
            std::vector<SE3> LastTryPoses = TheLastTryPoses(initialize_pose);
            for(auto& pose:LastTryPoses){
                TrackerStatus tmpstatus = TrackNewestFrameUsingCeres(target_frame,host_frame,pose);
                tmpstatus = CheckTrackStatus(target_frame,host_frame,pose,tmpstatus);
                if(tmpstatus != TrackerStatus::Lost){
                    pose_final_change = pose;
                    return tmpstatus;
                }
            }
        }else{
            return TestStatus;
        }
         */
        return CheckTrackStatus(target_frame,host_frame,TrackPose,trackStatus);
    }

    SE3 Tracker::CheckBestInitPose(Frame::Ptr& target_frame, const Frame::Ptr& host_frame,
                                   std::vector<SE3>& initialize_pose) {
        double min_cost = 9999,min_id;
        SE3 pose;
        for(size_t i = 0;i<initialize_pose.size();i++){
            double cost = TrackNearestError(target_frame,host_frame,initialize_pose[i]);
            //std::cout << "id: " << target_frame->Id_ << " cost: " << cost << " which: " << i << std::endl;
            if(min_cost > cost){
                min_cost = cost;
                min_id = i;

            }
        }
        pose = initialize_pose[min_id];
        LOG(INFO) << "Check the best initialization " << min_id << " Id: " << target_frame->Id_;
        return pose;
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
                Vec3 Pxyz_target = CameraConfig_->camera2camera(Pxyz_host,initialize_pose);
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

    double Tracker::ReprojectError(const EdgeVO::Frame::Ptr& target_frame, const EdgeVO::Frame::Ptr& host_frame,
                                   const SE3 &initialize_pose) {

        int good_num = 0;
        double error = 0;
        for(size_t lvl = 0;lvl < TrackConifg_->PyramidLevel_;++lvl){
            for(auto &pixel : host_frame->EdgePixels_[lvl]){
                Vec2 Puv_host = pixel->ReturnPixelPosition();
                double depth_host = pixel->Depth_;
                Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host,depth_host,lvl);
                Vec3 Pxyz_target = CameraConfig_->camera2camera(Pxyz_host,initialize_pose);
                Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target,lvl);

                if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0){
                    double e = (Puv_host - Puv_target).norm();
                    //if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                    //    continue;
                    //}

                    if(std::isnan(e)){
                        LOG(ERROR) << "Bad info " << Puv_target << CameraConfig_->SizeW_[lvl] << " " << CameraConfig_->SizeH_[lvl];
                        exit(0);
                    }
                    error += e;
                    ++good_num;
                }
            }
        }
        if(good_num == 0){
            LOG(ERROR) << "Reproject no good edge !" << " host id: " << host_frame->Id_ << " target id: " << target_frame->Id_ << std::endl;
            return MAXFLOAT;
        }

        return error / good_num;
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

    double Tracker::TrackNearestErrorOnlvl(const EdgeVO::Frame::Ptr &target_frame, const EdgeVO::Frame::Ptr &host_frame,
                                           const SE3 &initialize_pose, int lvl) {

        int good_num = 0;
        double error = 0;
        for(auto &pixel : host_frame->EdgePixels_[lvl]){
            Vec2 Puv_host = pixel->ReturnPixelPosition();
            double depth_host = pixel->Depth_;
            Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host,depth_host,lvl);
            Vec3 Pxyz_target = CameraConfig_->camera2camera(Pxyz_host,initialize_pose);
            Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target,lvl);

            if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0){
                int x = static_cast<int>(Puv_target[0]);
                int y = static_cast<int>(Puv_target[1]);
                Vec2 target_edge = target_frame->GetNearestEdge(x,y,lvl);
                if(!Utility::InBorder(target_edge,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_)){
                    continue;
                }

                Vec2 gradient = target_frame->GetGradient(target_edge,lvl);

                //if(gradient[0] == 0 && gradient[1] == 0) continue;

                Vec2 g = gradient.normalized();
                Vec2 error_Vec = Puv_target - target_edge;
                double e = error_Vec.dot(g);
                if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                    continue;
                }

                if(std::isnan(e)){
                    LOG(ERROR) << "Bad info " << error_Vec.transpose() << "  " << CameraConfig_->SizeW_[lvl] << " " << CameraConfig_->SizeH_[lvl];
                    exit(0);
                }
                error += e;
                ++good_num;
            }
        }
        if(good_num == 0){
            LOG(ERROR) << "Track no good edge !" << " host id: " << host_frame->Id_ << " target id: " << target_frame->Id_ << std::endl;
            return MAXFLOAT;
        }
        return error / good_num;
    }
    double Tracker::TrackNearestError(const EdgeVO::Frame::Ptr &target_frame, const EdgeVO::Frame::Ptr &host_frame,
                                      const SE3 &initialize_pose) {
        int good_num = 0;
        double error = 0;
        for(size_t lvl = 0;lvl < TrackConifg_->PyramidLevel_;++lvl){
            for(auto &pixel : host_frame->EdgePixels_[lvl]){
                Vec2 Puv_host = pixel->ReturnPixelPosition();
                double depth_host = pixel->Depth_;
                Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host,depth_host,lvl);
                Vec3 Pxyz_target = CameraConfig_->camera2camera(Pxyz_host,initialize_pose);
                Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target,lvl);

                if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0){
                    int x = static_cast<int>(Puv_target[0]);
                    int y = static_cast<int>(Puv_target[1]);
                    Vec2 target_edge = target_frame->GetNearestEdge(x,y,lvl);
                    if(!Utility::InBorder(target_edge,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_)){
                        continue;
                    }
                    Vec2 gradient = target_frame->GetGradient(target_edge,lvl);

                    //if(gradient[0] == 0 && gradient[1] == 0) continue;

                    Vec2 g = gradient.normalized();
                    Vec2 error_Vec = Puv_target - target_edge;
                    //double e = error_Vec.dot(error_Vec);
                    double e = error_Vec.dot(g);
                    if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                        continue;
                    }

                    if(std::isnan(e)){
                        LOG(ERROR) << "Bad info " << target_frame->Id_ << " lvl: "<< lvl << " puv_target: "<< x << " " << y << "  target_edge: " << target_edge.transpose() << "  error: " << error_Vec.transpose() << "  " << g.transpose();
                        exit(0);
                    }
                    error += e;
                    ++good_num;
                }
            }
        }
        if(good_num == 0){
            LOG(ERROR) << "Track no good edge !" << " host id: " << host_frame->Id_ << " target id: " << target_frame->Id_ << std::endl;
            return MAXFLOAT;
        }
        return error / good_num;
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
            if(lvl == 0){
                Debug(host_frame,target_frame,initialize_pose,lvl);
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
                if(abs(error) < abs(lastError)){
                    initialize_pose = T_t_h_new;
                    //std::cout  << "Increment accepted at iteration : "<<iter << " lambda: " << lambda << " error: " << error << " last error: " << lastError  << std::endl;
                    status = TrackerStatus ::Ok;
                    if(abs(error) / abs(lastError) > 0.999f){
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
        ls.initialize();
        size_t idx = 0;

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

            /*
            J[0] = fx_f / z;
            J[1] = fy_f / z;
            J[2] = (-1 / (z * z)) * (fx_f * x + fy_f * y);
            J[3] = -fx_f * x * y / (z*z)  - fy_f * (1 + (y*y) / (z*z));
            J[4] = fx_f * (1 + (x*x) / (z*z)) + fy_f * (x*y)/(z*z);
            J[5] = -fx_f * y / z + fy_f * x / z;
             */
            ls.update(J,buf_warped_residual[idx],buf_warped_weight[idx]);
        }
        ls.finish();
    }


    double Tracker::TrackNewestError(EdgeVO::Frame::Ptr &target_frame, const EdgeVO::Frame::Ptr &host_frame,
                                     SE3 &initialize_pose, int lvl) {

        Vec3* DTInfo_ = target_frame->PyramidDT_[lvl];

        mBufferInfo_->Reset();
        cv::Mat track_mask = cv::Mat::zeros(cv::Size(CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl]),CV_32SC1);

        std::vector<double> ErrorVec;
        for(auto &pixel : host_frame->EdgePixels_[lvl]){
            if(pixel->Depth_ < 0) continue;
            Vec2 Puv_host = pixel->ReturnPixelPosition();
            double depth_host = pixel->Depth_;
            Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host,depth_host,lvl);
            Vec3 Pxyz_target = CameraConfig_->camera2camera(Pxyz_host,initialize_pose);
            Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target,lvl);

            if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0) {
                int x = static_cast<int>(Puv_target[0]);
                int y = static_cast<int>(Puv_target[1]);
                Vec2 target_edge = target_frame->GetNearestEdge(x,y,lvl);
                if(!Utility::InBorder(target_edge,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_)){
                    continue;
                }
                /*
                if(track_mask.at<int>(y,x) == 0){
                    track_mask.at<int>(y,x) = 1;
                }else{
                    continue;
                }
                 */

                Vec2 gradient = target_frame->GetGradient(target_edge,lvl);

                Vec2 g = gradient.normalized();
                Vec2 error_Vec = Puv_target - target_edge;
                double e = error_Vec.dot(g);

                float huber_weight = Utility::GetHuberWeight(e,TrackConifg_->TrackHuberWeright_);

                double v = 2, theta = 1;

                float weight = (v + 1) / (v + pow(e / theta, 2));

                g = g;
                int filter[3] = {5,7,10};
                if(TrackConifg_->UseTrackFilter_ && e > filter[lvl]){
                    ++mBufferInfo_->BadEdgesNum_;
                    continue;
                }
                buf_warped_x[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(Pxyz_target[0]);
                buf_warped_y[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(Pxyz_target[1]);
                buf_warped_depth[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(Pxyz_target[2]);
                buf_warped_residual[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(e);
                buf_warped_dx[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(g[0]);
                buf_warped_dy[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(g[1]);
                buf_warped_weight[mBufferInfo_->GoodEdgesNum_] = static_cast<float>(weight);
                mBufferInfo_->SumError_ = mBufferInfo_->SumError_ + std::abs(e);
                ++mBufferInfo_->GoodEdgesNum_;
            } else{
                ++mBufferInfo_->BadEdgesNum_;
                continue;
            }

        }

        if(mBufferInfo_->GoodEdgesNum_ != 0){
            //std::cout << "Good num: " << mBufferInfo_->GoodEdgesNum_ << std::endl;
            return mBufferInfo_->SumError_ / static_cast<float>(mBufferInfo_->GoodEdgesNum_);
        }else{
            //std::cout << "Fuck num: " << mBufferInfo_->GoodEdgesNum_ << std::endl;
            return -MAXFLOAT;
        }
    }

    void Tracker::Debug(Frame::Ptr host,Frame::Ptr target,SE3 &pose,int lvl) {

        cv::Mat rgb = target->RGBImgs_[lvl];

        for(auto &pixel : host->EdgePixels_[lvl]) {
            Vec2 Puv_host = pixel->ReturnPixelPosition();
            double depth_host = pixel->Depth_;
            Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host, depth_host, lvl);
            Vec3 Pxyz_target = CameraConfig_->camera2camera(Pxyz_host, pose);
            Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target, lvl);

            if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_)){
                int x = static_cast<int>(Puv_target[0]);
                int y = static_cast<int>(Puv_target[1]);
                Vec2 target_edge = target->GetNearestEdge(x, y, lvl);
                //cv::line(rgb,cv::Point(Puv_target[0],Puv_target[1]),cv::Point(target_edge[0],target_edge[1]),cv::Scalar(0,255,0));
                cv::circle(rgb,cv::Point(Puv_target[0],Puv_target[1]),2,cv::Scalar(255,0,0));
                //cv::circle(rgb,cv::Point(target_edge[0],target_edge[1]),2,cv::Scalar(0,0,255));
            }

        }
        for(auto &pixel: target->EdgePixels_[lvl]){
            cv::circle(rgb,cv::Point(pixel->Hostx_,pixel->Hosty_),2,cv::Scalar(0,255,0));
        }

        cv::imshow("rgb" + std::to_string(target->Id_),rgb);
        cv::waitKey(10);
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
        if(status == Tracker::TrackerStatus::Lost){
            LOG(INFO) << "No Update ";
            return Tracker::TrackerStatus::Lost;
        }
        return CheckKeyFrame(target_frame,host_frame,initialize_pose);
    }

    Tracker::TrackerStatus Tracker::CheckKeyFrame(const EdgeVO::Frame::Ptr& target_frame,const EdgeVO::Frame::Ptr& host_frame,
                                                  const SE3 &initialize_pose) {

        double error = TrackNearestError(target_frame,host_frame,initialize_pose);
        mDebuger_->AllAfterError_ += error;
        DebugError(error);
        if(error > TrackConifg_->TrackMaxError_){
            LOG(INFO) << "Lost for big error! Id: " << target_frame->Id_ << " e: " <<  error;
            mDebuger_->LostNumAfterCheck_++;
            return TrackerStatus ::Lost;
        }

        /**相似性*/
        double track_move = ReprojectError(target_frame,host_frame,initialize_pose);
        if(track_move > TrackConifg_->TrackParallaxError_){
            mDebuger_->KeyNum_++;
            LOG(INFO) << "New KeyFrame with moving: " <<  track_move << " num: " << mBufferInfo_->GoodEdgesNum_;
            return TrackerStatus::NewKeyframe;
        }

        if(mBufferInfo_->GoodEdgesNum_ < mBufferInfo_->BadEdgesNum_ * 2) {
            LOG(INFO) << "KeyFrame for few good: " << "(g,b)  (" << mBufferInfo_->GoodEdgesNum_ << "," << mBufferInfo_->BadEdgesNum_ << ")";
            return TrackerStatus::NewKeyframe;
        }
        ++mDebuger_->GoodNum_;
        LOG(INFO) << "Just Ok! Id: " << target_frame->Id_ << " with error: " << error << " num: " << mBufferInfo_->GoodEdgesNum_;
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


