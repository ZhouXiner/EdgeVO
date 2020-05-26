//
// Created by zhouxin on 2020/5/26.
//

#ifndef EDGEVO_CONFIG_H
#define EDGEVO_CONFIG_H
#include "CommonInclude.h"

/**The system config load all the information needed*/
namespace EdgeVO{
    class SystemConfig{
    public:
        class Frame;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<SystemConfig> Ptr;

        SystemConfig(std::string config_path){
            ConfigInfo_ = YAML::LoadFile(config_path);
            SystemPath_ = ConfigInfo_["project_path"].as<std::string>();

            /**Image*/
            ImageFilePath_ = ConfigInfo_["image_dataset_info_path"].as<std::string>();
            ImageDatasetPath_ = ConfigInfo_["image_dataset_path"].as<std::string>();
            ImageDepthScale_ = ConfigInfo_["image_depth_scale"].as<int>();
            ImageWidth_ = ConfigInfo_["image_width"].as<int>();
            ImageHeight_ = ConfigInfo_["image_height"].as<int>();
            ImageDepthMax_ = ConfigInfo_["image_depth_max"].as<double>();
            ImageDepthMin_ = ConfigInfo_["image_depth_min"].as<double>();
            ImageBound_ = ConfigInfo_["image_border_size"].as<int>();

            /**Camera*/
            PyramidLevel_ = ConfigInfo_["pyramid_level"].as<int>();
            PyramidFactor_ = ConfigInfo_["pyramid_factor"].as<double>();
            fx = ConfigInfo_["IntrinsicsCamFx"].as<double>();
            fy = ConfigInfo_["IntrinsicsCamFy"].as<double>();
            cx = ConfigInfo_["IntrinsicsCamCx"].as<double>();
            cy = ConfigInfo_["IntrinsicsCamCy"].as<double>();
            k1 = ConfigInfo_["IntrinsicsCamK1"].as<double>();
            k2 = ConfigInfo_["IntrinsicsCamK2"].as<double>();
            k3 = ConfigInfo_["IntrinsicsCamK3"].as<double>();
            p1 = ConfigInfo_["IntrinsicsCamP1"].as<double>();
            p2 = ConfigInfo_["IntrinsicsCamP2"].as<double>();

            /**Frame*/
            CannyThresholdL_ = ConfigInfo_["cannythreshold_low"].as<int>();
            CannyThresholdH_ = ConfigInfo_["cannythreshold_high"].as<int>();
            EdgePatchSize_ = ConfigInfo_["edge_patch_num"].as<int>();
            UseEdgePatch_ = ConfigInfo_["use_edge_balance"].as<bool>();

            /**System Info*/
            InitMinNum_ = ConfigInfo_["initialize_min_num"].as<int>();
            OutputPath_ = ConfigInfo_["output_path"].as<std::string>();

            /**Track info*/
            TrackIters_ = ConfigInfo_["track_iters"].as<int>();
            TrackSolverTime_ = ConfigInfo_["track_solver_time"].as<double>();
            TrackMaxError_ = ConfigInfo_["track_max_error_after_track"].as<double>();
            TrackParallaxError_ = ConfigInfo_["track_parallax_error"].as<double>();
            TrackHuberWeright_ = ConfigInfo_["track_huber_weight"].as<double>();
            TrackLeastNum_.resize(PyramidLevel_);
            TrackMostNum_.resize(PyramidLevel_);
            TrackFilter_.resize(PyramidLevel_);
            TrackLeastNum_[0] = ConfigInfo_["track_edge_least_num"]["lvl0"].as<int>();
            TrackLeastNum_[1] = ConfigInfo_["track_edge_least_num"]["lvl1"].as<int>();
            TrackLeastNum_[2] = ConfigInfo_["track_edge_least_num"]["lvl2"].as<int>();
            TrackMostNum_[0] = ConfigInfo_["track_edge_most_num"]["lvl0"].as<int>();
            TrackMostNum_[1] = ConfigInfo_["track_edge_most_num"]["lvl1"].as<int>();
            TrackMostNum_[2] = ConfigInfo_["track_edge_most_num"]["lvl2"].as<int>();
            TrackFilter_[0] = ConfigInfo_["track_error_filter"]["lvl0"].as<double>();
            TrackFilter_[1] = ConfigInfo_["track_error_filter"]["lvl1"].as<double>();
            TrackFilter_[2] = ConfigInfo_["track_error_filter"]["lvl2"].as<double>();

            /**Local Mapper Info*/
            LMEnable_ = ConfigInfo_["lm_enable"].as<bool>();
            LMInitMinNum_ = ConfigInfo_["lm_min_kf_num"].as<int>();
            LMWindowSize_ = ConfigInfo_["lm_max_kf_num"].as<int>();
            LMMaxRange_ = ConfigInfo_["lm_max_range"].as<int>();
            LMGoodEdgeDTThreshold_ = ConfigInfo_["lm_good_edge_dt_threshold"].as<float>();
            LMGoodEdgeDistanceThreshold_ = ConfigInfo_["lm_good_edge_distancemap_threshold"].as<float>();
            LMEdgeObservationThreshold_ = ConfigInfo_["lm_edge_observation_threshold"].as<int>();
            LMFeatureSize_ = ConfigInfo_["lm_feature_size"].as<int>();
            LMMargeSize_ = ConfigInfo_["lm_marge_size"].as<int>();

            /**Loop*/
            LoopMinDistance_ = ConfigInfo_["loop_min_kf_distance"].as<int>();
            LoopMinDistanceBetweenTwoLoop_ = ConfigInfo_["loop_min_between_distance"].as<int>();
            LoopMinDistanceToLastNFrame_ = ConfigInfo_["loop_min_lastN_distance"].as<int>();
            LoopMinDistanceNewDBForLoop_ = ConfigInfo_["loop_min_newdb_forloop_distance"].as<int>();
            LoopMinSimilirity_ = ConfigInfo_["loop_min_similirity"].as<float>();
            LoopMinSimilirityDB_ = ConfigInfo_["loop_min_similirity_for_newDB"].as<float>();
            LoopQualityCheckRange_ = ConfigInfo_["loop_quality_check_range"].as<int>();
            LoopGraphBAConstraints_ = ConfigInfo_["loop_graph_constraint_num"].as<int>();
            LoopHistogramDis_ = ConfigInfo_["loop_histogram_dis"].as<float>();
            LoopDebug_ = ConfigInfo_["loop_debug"].as<bool>();
            LoopOnlyFernKFForLMConstrain_ = ConfigInfo_["loop_only_fernkf_for_lm_constraint"].as<bool>();
            LoopConstraintWeight_ = ConfigInfo_["loop_loop_constraint_weight"].as<int>();

            /**Relocalisation*/
            RLRemoveNum_ = ConfigInfo_["rl_remove_n_num"].as<int>();
            RLMinKFNum_ = ConfigInfo_["rl_min_kf_num"].as<int>();

            /**Fern DataBase*/
            FernEnable_ = ConfigInfo_["fern_enable"].as<bool>();
            FernMaxNum_ = ConfigInfo_["fern_database_maxnum"].as<int>();
            FernCodeNum_ = ConfigInfo_["fern_code_num"].as<int>();
            FernLevel_ = ConfigInfo_["fern_level"].as<int>();
            FernNumPreFern_ = ConfigInfo_["fern_num_prefern"].as<int>();
            FernSizePreCode_ = (1 << FernNumPreFern_);
            FernSimilarityThreshold_ = ConfigInfo_["fern_similarity_threshold"].as<double>();
            FernUseDepth_ = ConfigInfo_["fern_use_depth"].as<bool>();
            FernUseDT_ = ConfigInfo_["fern_use_dt"].as<bool>();
            DefineRGBBound_ = Vec2f(0,255);
            DefineDepthBound_;
            DefineDTBound_;
        };

        /**Project Info*/
        std::string SystemPath_;

        /**Image Info*/
        std::string ImageFilePath_;
        std::string ImageDatasetPath_;
        int ImageDepthScale_;
        int ImageWidth_;
        int ImageHeight_;
        double ImageDepthMin_;
        double ImageDepthMax_;
        double ImageBound_;

        /**Camera Info*/
        double fx,fy,cx,cy,k1,k2,k3,p1,p2;
        int PyramidLevel_ = 3;
        double PyramidFactor_;

        /**Frame Info*/
        int CannyThresholdL_;
        int CannyThresholdH_;
        int EdgePatchSize_;
        bool UseEdgePatch_;

        /**System Info*/
        int InitMinNum_;
        std::string OutputPath_;

        /**Track info*/
        std::vector<double> TrackFilter_;
        int TrackIters_;
        double TrackSolverTime_;
        double TrackMaxError_;
        double TrackHuberWeright_;
        bool UseTrackFilter_ = true;
        double TrackParallaxError_;
        std::vector<int> TrackLeastNum_;
        std::vector<int> TrackMostNum_;

        /**Local Mapper Info*/
        bool LMEnable_;
        int LMInitMinNum_;
        int LMWindowSize_;
        float LMGoodEdgeDTThreshold_;
        float LMGoodEdgeDistanceThreshold_;
        int LMEdgeObservationThreshold_;
        int LMFeatureSize_;
        int LMMargeSize_;
        int LMMaxRange_;

        /**Loop Info*/
        int LoopMinDistance_;
        float LoopMinSimilirity_;
        int LoopQualityCheckRange_;
        float LoopHistogramDis_;
        bool LoopDebug_;
        float LoopMinSimilirityDB_;
        int LoopMinDistanceBetweenTwoLoop_;
        int LoopMinDistanceNewDBForLoop_;
        int LoopGraphBAConstraints_;
        int LoopMinDistanceToLastNFrame_;
        bool LoopOnlyFernKFForLMConstrain_;
        int LoopConstraintWeight_;
        double LoopHistWeights_[4] = {1.0,1.0,1.25,1.5};

        /**Relocolisation*/
        int RLRemoveNum_;
        int RLMinKFNum_;
        /**Fern DataBase Info*/
        bool FernEnable_;
        int FernMaxNum_;
        int FernCodeNum_;
        int FernLevel_;
        int FernNumPreFern_;
        double FernSimilarityThreshold_ = 0.05;
        int FernSizePreCode_;
        bool FernUseDepth_;
        bool FernUseDT_;
        Vec2f DefineRGBBound_;
        Vec2f DefineDepthBound_;
        Vec2f DefineDTBound_;

    private:
        YAML::Node ConfigInfo_;
    };
}
#endif //EDGEVO_CONFIG_H
