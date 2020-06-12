//
// Created by zhouxin on 2020/3/26.
//
#include "EdgeVO/Track.h"

namespace EdgeVO{
    Tracker::Tracker(const SystemConfig::Ptr system_config,const CameraConfig::Ptr camera_config) {
        TrackConifg_ = system_config;
        CameraConfig_ = camera_config;
        se3_pose_double = (double*)malloc(sizeof(double) * 6);
    }

    Tracker::~Tracker(){
        free(se3_pose_double);
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
        TrackerStatus trackStatus = TrackNewestFrameUsingCeres(target_frame,host_frame,TrackPose);

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
                    if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                        continue;
                    }

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
                    Vec2 gradient = target_frame->GetGradient(Puv_target,lvl);

                    if(gradient[0] == 0 && gradient[1] == 0) continue;

                    Vec2 g = gradient.normalized();
                    Vec2 error_Vec = Puv_target - target_edge;
                    //double e = error_Vec.dot(error_Vec);
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
        }
        if(good_num == 0){
            LOG(ERROR) << "Track no good edge !" << " host id: " << host_frame->Id_ << " target id: " << target_frame->Id_ << std::endl;
            return MAXFLOAT;
        }
        return error / good_num;
    }
    Tracker::TrackerStatus Tracker::TrackNewestFrameUsingCeres(EdgeVO::Frame::Ptr &target_frame,
                                                               const EdgeVO::Frame::Ptr &host_frame,
                                                               SE3 &initialize_pose) {
        SE3 back_pose = initialize_pose;
        TrackerStatus trackStatus_final = TrackerStatus::Lost;
        TrackerStatus trackStatus;

        //Debug(host_frame,target_frame,0);

        for(int lvl = TrackConifg_->PyramidLevel_ - 1;lvl >= 0;--lvl){
            if(host_frame->EdgePixels_[lvl].size() < TrackConifg_->TrackLeastNum_[lvl]){
                LOG(WARNING) << "Too Few Edges in Frame: " << target_frame->Id_  << " Lvl: " << lvl << " Num: " << host_frame->EdgePixels_[lvl].size();
                continue;
            }

            back_pose = initialize_pose;
            trackStatus = TrackNewestFrameUsingCeresOnLvl(target_frame,host_frame,initialize_pose,lvl);
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

        return trackStatus_final;
    }

    Tracker::TrackerStatus Tracker::TrackNewestFrameUsingCeresOnLvl(EdgeVO::Frame::Ptr &target_frame,
                                                                    const EdgeVO::Frame::Ptr &host_frame,
                                                                    SE3 &initialize_pose, int lvl) {
        mBufferInfo_->Reset();
        double cost_before = TrackNearestError(target_frame,host_frame,initialize_pose);

        ceres::Problem problem;
        ceres::LossFunction* loss_function;
        //loss_function = NULL;
        loss_function = new ceres::CauchyLoss(1.0);
        //loss_function = new ceres::HuberLoss(1.0);

        se3_pose_vector = initialize_pose.log();
        vector2double();

        ceres::LocalParameterization *localParamter = new PoseLocalParameter();
        problem.AddParameterBlock(se3_pose_double, 6, localParamter);

        for(auto &pixel : host_frame->EdgePixels_[lvl]) {
            Vec2 Puv_host = pixel->ReturnPixelPosition();
            double depth_host = pixel->Depth_;
            Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host, depth_host, lvl);
            Vec3 Pxyz_target = CameraConfig_->camera2camera(Pxyz_host, initialize_pose);
            Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target, lvl);

            if(Utility::InBorder(Puv_target,CameraConfig_->SizeW_[lvl],CameraConfig_->SizeH_[lvl],CameraConfig_->ImageBound_) && Pxyz_target[2] > 0) {

                int x = static_cast<int>(Puv_target[0]);
                int y = static_cast<int>(Puv_target[1]);
                Vec2 target_edge = target_frame->GetNearestEdge(x,y,lvl);
                double e = (Puv_target - target_edge).norm();

                if(TrackConifg_->UseTrackFilter_ && e > TrackConifg_->TrackFilter_[lvl]){
                    //std::cout << Puv_target.transpose() << " " << target_edge.transpose() << " " << Puv_host.transpose() << std::endl;
                    ++mBufferInfo_->BadEdgesNum_;
                    continue;
                }

                Vec2 gradient = target_frame->GetGradient(Puv_target,lvl);
                if(gradient[0] == 0 && gradient[1] == 0) continue;
                TrackCostFunction* trackcostfactor = new TrackCostFunction(Puv_host,gradient,depth_host,target_frame,CameraConfig_,lvl);
                problem.AddResidualBlock(trackcostfactor,loss_function,se3_pose_double);
                ++mBufferInfo_->GoodEdgesNum_;
            } else{
                ++mBufferInfo_->BadEdgesNum_;
                continue;
            }
        }

        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::DOGLEG;
        //options.max_solver_time_in_seconds = 0.03;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";

        double2vector();

        initialize_pose = SE3::exp(se3_pose_vector);
        double cost_after = TrackNearestError(target_frame,host_frame,initialize_pose);

        if(abs(cost_after) < abs(cost_before)){
            LOG(INFO) << "Good Id: "<< target_frame->Id_ << " host: " << host_frame->Id_ << " Num: " << mBufferInfo_->GoodEdgesNum_
                      << " (Before,after) (" << cost_before << "," << cost_after << ")";
            return TrackerStatus::Ok;
        } else{
            LOG(INFO) << "Id: "<< target_frame->Id_ << " host: " << host_frame->Id_ << " Num: " << mBufferInfo_->GoodEdgesNum_
                      << " (Before,after) (" << cost_before << "," << cost_after << ")";
            return TrackerStatus::Ok;
        }
    }

    void Tracker::vector2double() {
        se3_pose_double[0] = se3_pose_vector[0];
        se3_pose_double[1] = se3_pose_vector[1];
        se3_pose_double[2] = se3_pose_vector[2];
        se3_pose_double[3] = se3_pose_vector[3];
        se3_pose_double[4] = se3_pose_vector[4];
        se3_pose_double[5] = se3_pose_vector[5];
    }

    void Tracker::double2vector() {
        se3_pose_vector[0] = se3_pose_double[0];
        se3_pose_vector[1] = se3_pose_double[1];
        se3_pose_vector[2] = se3_pose_double[2];
        se3_pose_vector[3] = se3_pose_double[3];
        se3_pose_vector[4] = se3_pose_double[4];
        se3_pose_vector[5] = se3_pose_double[5];
    }

    void Tracker::Debug(Frame::Ptr host,Frame::Ptr target,int lvl) {

        cv::Mat rgb = target->RGBImgs_[lvl];
        for(auto &pixel : host->EdgePixels_[lvl]) {
            Vec2 Puv_host = pixel->ReturnPixelPosition();
            double depth_host = pixel->Depth_;
            Vec3 Pxyz_host = CameraConfig_->pixel2camera(Puv_host, depth_host, lvl);
            Vec3 Pxyz_target = CameraConfig_->camera2camera(Pxyz_host, SE3());
            Vec2 Puv_target = CameraConfig_->camera2pixel(Pxyz_target, lvl);

            int x = static_cast<int>(Puv_target[0]);
            int y = static_cast<int>(Puv_target[1]);
            Vec2 target_edge = target->GetNearestEdge(x, y, lvl);
            //cv::line(rgb,cv::Point(Puv_target[0],Puv_target[1]),cv::Point(target_edge[0],target_edge[1]),cv::Scalar(0,255,0));
            cv::circle(rgb,cv::Point(Puv_target[0],Puv_target[1]),2,cv::Scalar(255,0,0));
            cv::circle(rgb,cv::Point(target_edge[0],target_edge[1]),2,cv::Scalar(0,0,255));
        }
        cv::imshow("rgb",rgb);
        cv::waitKey(0);
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

        double final_e = ReprojectError(target_frame,host_frame,initialize_pose);;
        mDebuger_->AllAfterError_ += final_e;
        DebugError(final_e);
        if(final_e > TrackConifg_->TrackMaxError_){
            LOG(INFO) << "Lost for big error! Id: " << target_frame->Id_ << " e: " <<  final_e;
            mDebuger_->LostNumAfterCheck_++;
            return TrackerStatus ::Lost;
        }

        /**相似性*/
        if(final_e > TrackConifg_->TrackParallaxError_){
            mDebuger_->KeyNum_++;
            LOG(INFO) << "New KeyFrame with " <<  final_e << " error " << " num: " << mBufferInfo_->GoodEdgesNum_;
            return TrackerStatus::NewKeyframe;
        }

        if(mBufferInfo_->GoodEdgesNum_ < mBufferInfo_->BadEdgesNum_ * 2) {
            LOG(INFO) << "KeyFrame for few good: " << "(g,b)  (" << mBufferInfo_->GoodEdgesNum_ << "," << mBufferInfo_->BadEdgesNum_ << ")";
            return TrackerStatus::NewKeyframe;
        }
        ++mDebuger_->GoodNum_;
        LOG(INFO) << "Just Ok! Id: " << target_frame->Id_ << " with error: " << final_e << " num: " << mBufferInfo_->GoodEdgesNum_;
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


