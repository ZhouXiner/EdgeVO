#include <iostream>
#include <string>
#include <gflags/gflags.h>
#include "EdgeVO/Config.h"
#include "EdgeVO/Inputs.h"
#include "EdgeVO/System.h"

int main(int argc, char* argv[]) {
    std::string ConfigPath_;
    switch(argc){
        case 0:
            ConfigPath_ = "/home/zhouxin/Github/EdgeVO/config/config.yaml";
            break;
        case 1:
            ConfigPath_ = "/home/zhouxin/Github/EdgeVO/config/config.yaml";
            break;
        case 2:
            ConfigPath_ = argv[1];
            break;
        default:
            std::cout << "Bad input !" << std::endl;
            return 0;
    }

    google::ParseCommandLineFlags(&argc, &argv, true);
    //google::InitGoogleLogging(argv[0]);
    //FLAGS_log_dir = "../log";
    EdgeVO::System::Ptr system(new EdgeVO::System(ConfigPath_));
    LOG(INFO) << "System Running!";
    system->Run();
    LOG(INFO) << "System End!";
    return 0;
}