//
// Created by zhouxin on 2020/4/14.
//

#ifndef IESLAM_UTILITY_H
#define IESLAM_UTILITY_H
#include <Eigen/Dense>

class Utility{
public:
    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
    {
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -q(2), q(1),
                q(2), typename Derived::Scalar(0), -q(0),
                -q(1), q(0), typename Derived::Scalar(0);
        return ans;
    }

    static double GetHuberWeight(double e, double huber) {
        //return (e<=huber ? 1 : pow(huber / e,2));
        return (e<=huber ? 1 : huber / e);

    }
    static double GetTDistributionWeight(double e, double theta) {
        return 6.0f / (5.0f + pow(e/theta,2));
    }

    static double GetTDistributionNewWeight(double e) {
        double v = 3,theta = 2;
        return (v + 1) / (v + pow(e / theta,2));
    }

    static bool InBorder(Eigen::Vector2d p,int w,int h,int bound){
        return (static_cast<int>(p[0]) >= bound && static_cast<int>(p[0]) <= (w - bound - 1)
                && static_cast<int>(p[1]) >= bound && static_cast<int>(p[1]) <= (h - bound - 1));
    }

    static float calcEuclideanDistance(int x1, int y1, int x2, int y2)
    {
        return sqrt(float((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)));
    }

    static bool InBorder(int u,int v,int w,int h,int bound){
        return (static_cast<int>(u) >= bound && static_cast<int>(u) <= (w - bound - 1)
               && static_cast<int>(v) >= bound && static_cast<int>(v) <= (h - bound - 1));
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
    {
        //printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
        //Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
        //printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
        //return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
        return q;
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }
    static float rand01()
    {
        return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
};
#endif //IESLAM_UTILITY_H
