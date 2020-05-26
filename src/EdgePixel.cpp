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

    void EdgePixel::UpdateObservation(int id){
        for(auto &ob:Observations_){
            if(ob > id){
                --ob;
            }
        }
        return;
    }

    bool EdgePixel::ChangeObservation(int id,int newest_id) {
        auto iteration = find(Observations_.begin(),Observations_.end(),id);

        if(iteration != Observations_.end()){
            //先删除，后调整id,为了确保不会将更改后的id删除
            Observations_.erase(iteration);
            --UsedNum_;

            UpdateObservation(id);
            if(Observations_.back() != newest_id){
                EdgeStatus_ = EdgePixel::EdgeStatus::Marged;
                return true;
            }
        } else{
            UpdateObservation(id);
        }
        return false;
    }

    bool EdgePixel::CheckGoodState(int obs) {
        if(EdgeStatus_ == EdgePixel::EdgeStatus::Good && Observations_.size() < obs){
            EdgeStatus_ = EdgePixel::EdgeStatus::Uncertain;
            return false;
        }
        return true;
    }
    bool EdgePixel::ChooseGoodEdge(int obs) {
        if(EdgeStatus_ == EdgePixel::EdgeStatus::Good && Observations_.size() >= obs){
            return true;
        }
        return false;
    }
    bool EdgePixel::ChooseNewGoodEdge(int obs) {
        if(EdgeStatus_ == EdgePixel::EdgeStatus::Uncertain && Observations_.size() >= obs){
            return true;
        }
        return false;
    }
}
