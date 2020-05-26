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
        if(CurrentFrame_->IsInDB()){
            InDB_.push_back(true);
        }else{
            InDB_.push_back(false);
        }
        if(CurrentFrame_->IsKeyFrame_){
            IsKeyFrame_.push_back(true);
        }else{
            IsKeyFrame_.push_back(false);
        }
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

        const float blue[3] = {0, 0, 1};
        const float green[3] = {0, 1, 0};

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
        cv::Mat img_out = CurrentFrame_->GrayImgs_[0];
        int edge_size = CurrentFrame_->EdgePixels_[0].size();
        cv::putText(img_out,std::to_string(CurrentFrame_->Id_),cv::Point(100,100),1,1,cv::Scalar(255,0,0),2);
        for (auto &pixel:CurrentFrame_->EdgePixels_[0]) {
            Vec2 position = pixel->ReturnPixelPosition();
                cv::circle(img_out, cv::Point2f(position[0],position[1]), 2, cv::Scalar(0, 250, 0),
                           2);
        }
        return img_out;
    }

    void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
        SE3 Twc = CurrentFrame_->Tcw_.inverse();
        pangolin::OpenGlMatrix m(Twc.matrix());
        vis_camera.Follow(m, true);
    }

    void Viewer::DrawTrajectory() {

        for (size_t i = 0; i < ActiveKeyFrames_.size() - 1; i++) {

            if(InDB_[i]){
                glColor3f(1.0, 0.0, 0.0);
            }else{
                glColor3f(0.0, 0.0, 0.0);
            }
            if(IsKeyFrame_[i]){
                glLineWidth(4);
            }else{
                glLineWidth(2);
            }
            glBegin(GL_LINES);
            Vec3 p1 = ActiveKeyFrames_[i]->Tcw_.inverse().translation();
            Vec3 p2 = ActiveKeyFrames_[i+1]->Tcw_.inverse().translation();
            glVertex3d(p1[0], p1[1], p1[2]);
            glVertex3d(p2[0], p2[1], p2[2]);

            glEnd();
        }

    }

    void Viewer::DrawFrame(Frame::Ptr frame, const float* color) {
        SE3 Twc = frame->Tcw_.inverse();
        const float sz = 1.0;
        const int line_width = 2.0;
        const float fx = 517.30640;
        const float fy = 516.469215;
        const float cx = 318.643040;
        const float cy = 255.313989;
        const float width = 640;
        const float height = 480;

        glPushMatrix();

        Sophus::Matrix4f m = Twc.matrix().template cast<float>();
        glMultMatrixf((GLfloat*)m.data());

        if (color == nullptr) {
            glColor3f(1, 0, 0);
        } else
            glColor3f(color[0], color[1], color[2]);

        glLineWidth(line_width);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glEnd();
        glPopMatrix();
    }

}
















