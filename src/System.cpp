//
// Created by zhouxin on 2020/3/24.
//
#include "EdgeVO/System.h"
namespace EdgeVO{

    System::System(){}
    System::System(std::string config_path) {
        SystemConfig_ = std::make_shared<SystemConfig>(config_path);
        CameraConfig_ = std::make_shared<CameraConfig>(SystemConfig_); /**For the Frame Tracking*/
        mFrameDataset_ = std::make_unique<FrameDataset>(SystemConfig_,CameraConfig_);
        mTracker_ = std::make_unique<Tracker>(SystemConfig_,CameraConfig_);
        mBackEnd_ = std::make_shared<BackEnd>(SystemConfig_,CameraConfig_);
        mViewer_ = std::make_unique<Viewer>();
       // mViewer_->SetMap(mLocalMapper_);
        //TrackOut_.open(SystemConfig_->OutputPath_,std::ofstream::out);
    }

    void System::Run() {
        mFrameDataset_->Run();
        while(mFrameDataset_->IsActive()){
            /**read  and handle a image using around 1e-4ms*/
            //auto t1=std::chrono::steady_clock::now();


            Frame::Ptr NewestFrame = mFrameDataset_->GetNewestFrame();

            if(NewestFrame == nullptr){
                std::this_thread::sleep_for(std::chrono::microseconds(5));
                continue;
            }
            //std::cout << "Id: " << NewestFrame->Id_ << " count: "<< NewestFrame.use_count() << std::endl;
            //auto t2=std::chrono::steady_clock::now();

            Tracking(NewestFrame);


            //double dr_ms=std::chrono::duration<double,std::milli>(t2-t1).count();

            //std::cout << "Read The Image cost: " << dr_ms << std::endl;
        }

        RecordAllTrajectory();
        auto debuger = mTracker_->mDebuger_;
        std::cout << "Max e: " << debuger->MaxError_ << " Min e: " << debuger->MinError_ <<  std::endl;
        std::cout << "Before all e: " << debuger->AllBeforeError_ << " After all e: " << debuger->AllAfterError_ << std::endl;
        std::cout << "Lost Num ceres: " << debuger->LostNumAfterCeres_ << " Lost Num check: " << debuger->LostNumAfterCheck_ << std::endl;
        std::cout << "Good: " << debuger->GoodNum_ << " Key: "<< debuger->KeyNum_ <<  " all num: " << debuger->FrameCount_ << std::endl;

        mViewer_->Close();
        //exit(0);

    }

    void System::Tracking(Frame::Ptr& newestFrame) {
        //std::cout << "traking start" << std::endl;
        switch(mSystemStatus_){
            case SystemStatus::Init :{

                mBackEnd_->AddNewestKeyFrame(newestFrame);
                mSystemStatus_ = SystemStatus::Tracking;
                break;
            }
            case SystemStatus::Tracking :{
                {
                   TrackingNewestFrame(newestFrame);
                }
                break;
            }
            case SystemStatus::Relocalisation :{
                ///if tracking lost, we will change the status and start relocalisation
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(TrajectoryMutex_);
            RecordSingleTrajectory(newestFrame);
            //std::cout << "id: " << newestFrame->Id_ << std::endl;
        }

        if(!LostInLastFrame_){
            mViewer_->AddCurrentFrame(newestFrame);
        }
        //std::cout << "traking end" << '\n' << std::endl;

    }

    void System::TrackingNewestFrame(EdgeVO::Frame::Ptr& newestFrame) {
        SE3 pose_t_h;
        Frame::Ptr host_frame = mBackEnd_->GetNewestKeyFrame();
        std::vector<SE3> TryPose = mBackEnd_->GetTryInitiPose();

        auto t1=std::chrono::steady_clock::now();
        Tracker::TrackerStatus track_status = mTracker_->TrackNewestFrame(newestFrame,host_frame,TryPose,pose_t_h);
        {
            std::lock_guard<std::mutex> lock(TrajectoryMutex_);
            newestFrame->UpdatePoseCW(pose_t_h * host_frame->ReturnPoseCW());
            newestFrame->UpdatePoseC1C2(pose_t_h);
            newestFrame->UpdateRefKF(host_frame);
        }

        auto t2=std::chrono::steady_clock::now();
        double dr_ms=std::chrono::duration<double,std::milli>(t2-t1).count();

        switch(track_status){
            case Tracker::TrackerStatus::Lost :{
                /**start relocolization restart or relocolization*/
                /**when lost, we will use the relocolisation mode to optimize the frame*/
                ///when lost, you can just add the frame or you can use some better idea
                mBackEnd_->AddNewestFrame(newestFrame);
                break;
            }
            case  Tracker::TrackerStatus::NewKeyframe :{
                {
                    std::lock_guard<std::mutex> lock(TrajectoryMutex_);
                    mBackEnd_->AddNewestKeyFrame(newestFrame);
                    break;
                }
            }
            case Tracker::TrackerStatus::Ok :{
                mBackEnd_->AddNewestFrame(newestFrame);
                break;
            }
            default:
                break;
        }
    }

    bool System::CheckRelocalisationMode(Frame::Ptr& newestFrame) {
        if(mBackEnd_->Reset()){
            LOG(INFO) << "Initilization restart! Id: " << newestFrame->Id_;
            mSystemStatus_ = SystemStatus::Init;
        }else{
            LOG(INFO) << "Relocalisation start! Id: " << newestFrame->Id_;
            mSystemStatus_ = SystemStatus::Relocalisation;
            //Relocalisation(newestFrame);
        }
        return true;
    }

    void System::RecordAllTrajectory() {
        for(auto &newestFrame:mBackEnd_->AllFrames_){
            SE3 Twc = newestFrame->Tcw_.inverse();
            Vec3 t = Twc.translation();
            Eigen::Quaterniond q(Twc.rotationMatrix());

            TrackOut_.open(SystemConfig_->OutputPath_, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
            TrackOut_ << std::fixed  << newestFrame->DepthTimeStamp_ << " " << std::setprecision(9) << " " << t[0] << " " << t[1] << " " << t[2]
                      << " " << q.x() << " " << q.y() << " " << q.z() << " " <<  q.w() << "\n";
            TrackOut_.close();
        }
    }

    void System::RecordSingleTrajectory(EdgeVO::Frame::Ptr &newestKF) {

        SE3 Twc = newestKF->Tcw_.inverse();
        Vec3 t = Twc.translation();
        Eigen::Quaterniond q(Twc.rotationMatrix());
        SingleTrackOut_.open("/home/zhouxin/GitHub/EdgeVO/output/Single_track.txt", std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
        SingleTrackOut_ << std::fixed  << newestKF->DepthTimeStamp_ << " " << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2]
        << " " << q.x() << " " << q.y() << " " << q.z() << " " <<  q.w() << "\n";
        SingleTrackOut_.close();

    }
    void System::DebugShowRGBImage(EdgeVO::Frame::Ptr& frame) {

        std::cout << "Id: "  << frame->Id_ <<  " size: " << frame->EdgePixels_[0].size() <<  " " << frame->EdgePixels_[1].size() << " " << frame->EdgePixels_[2].size() << std::endl;
        cv::Mat test1 = frame->RGBImgs_[0];
        for (auto &pixel:frame->EdgePixels_[0]) {
            Vec2 position = pixel->ReturnPixelPosition();
            cv::circle(test1, cv::Point2f(position[0],position[1]), 0.5, cv::Scalar(0, 250, 0),
                       2);
        }

        cv::Mat test2 = frame->RGBImgs_[1];
        for (auto &pixel:frame->EdgePixels_[0]) {
            Vec2 position = pixel->ReturnPixelPosition();
            cv::circle(test2, cv::Point2f(position[0],position[1]), 0.5, cv::Scalar(0, 250, 0),
                       2);
        }

        cv::Mat test3 = frame->RGBImgs_[2];
        for (auto &pixel:frame->EdgePixels_[0]) {
            Vec2 position = pixel->ReturnPixelPosition();
            cv::circle(test3, cv::Point2f(position[0],position[1]), 0.5, cv::Scalar(0, 250, 0),
                       2);
        }

        std::string title = std::to_string(frame->RGBTimeStamp_);
        cv::imshow(title,test1);
        cv::imshow("test1",test2);
        cv::imshow("test3",test3);
        cv::waitKey(0);
    }
}

