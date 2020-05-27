//
// Created by zhouxin on 2020/4/1.
//
#include "EdgeVO/Viewer.h"
#include "EdgeVO/EdgePixel.h"
namespace EdgeVO{
    Viewer::Viewer(){
        ViewerThread_ = std::thread(std::bind(&Viewer::ThreadLoop,this));
    };

    void Viewer::Close() {
        ViewerRunning_ = false;
        ViewerThread_.join();
    }

    void Viewer::AddCurrentFrame(const Frame::Ptr& current_frame) {
        std::lock_guard<std::mutex> lock(ViewerDataMutex_);
        CurrentFrame_ = current_frame;
        ActiveKeyFrames_.push_back(CurrentFrame_);
        AddCurrentPose();
    }
    void Viewer::AddCurrentPose() {
        SE3 Twc = CurrentFrame_->Tcw_.inverse();
        Vec3 translation = Twc.translation();
        Translations_.push_back(translation);
    }

    void Viewer::ThreadLoop() {
        pangolin::CreateWindowAndBind("IESLAM", 640, 480);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState vis_camera(
                pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.1, 1000),
                pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, pangolin::AxisNegY));

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& vis_display =
                pangolin::CreateDisplay()
                        .SetBounds(0.0, 1.0, 0, 1.0, -640.0f / 480.0f)
                        .SetHandler(new pangolin::Handler3D(vis_camera));

        while (!pangolin::ShouldQuit() && ViewerRunning_) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            vis_display.Activate(vis_camera);

            std::lock_guard<std::mutex> lock(ViewerDataMutex_);

            if(CurrentFrame_){
                //DrawFrame(CurrentFrame_, green);
                DrawTrajectory();
                //FollowCurrentFrame(vis_camera);

                cv::Mat img = PlotFrameImage();
                cv::imshow("image", img);
                cv::waitKey(1);
            }

            pangolin::FinishFrame();
            usleep(5000);
        }

        LOG(INFO) << "Stop viewer";
    }

    cv::Mat Viewer::PlotFrameImage() {
        cv::Mat img_out = CurrentFrame_->RGBImgs_[0];
        int edge_size = CurrentFrame_->EdgePixels_[0].size();
        cv::putText(img_out,std::to_string(CurrentFrame_->Id_),cv::Point(100,100),1,3,cv::Scalar(255,0,0),2);
        for (auto &pixel:CurrentFrame_->EdgePixels_[0]) {
            Vec2 position = pixel->ReturnPixelPosition();
                cv::circle(img_out, cv::Point2f(position[0],position[1]), 2, cv::Scalar(0, 250, 0),
                           1);
        }
        return img_out;
    }

    void Viewer::DrawTrajectory() {
        for (size_t i = 0; i < ActiveKeyFrames_.size() - 1; i++) {
            glBegin(GL_LINES);
            glLineWidth(3);
            glColor3f(0.0, 0.0, 0.0);
            Vec3 p1 = ActiveKeyFrames_[i]->Tcw_.inverse().translation();
            Vec3 p2 = ActiveKeyFrames_[i+1]->Tcw_.inverse().translation();
            glVertex3d(p1[0], p1[1], p1[2]);
            glVertex3d(p2[0], p2[1], p2[2]);
            glEnd();
        }

    }
}
















