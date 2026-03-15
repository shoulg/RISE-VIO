#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <deque>
#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
// Eigen typedefs matrices and vectors

// Eigen typedefs matrices and vectors
namespace Eigen {
template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;
template <typename T>
using aligned_list = std::list<T, Eigen::aligned_allocator<T>>;
template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename K, typename V>
using aligned_map = std::map<K, V, std::less<K>,
                             Eigen::aligned_allocator<std::pair<K const, V>>>;

template <typename K, typename V>
using aligned_unordered_map =
std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                   Eigen::aligned_allocator<std::pair<K const, V>>>;
} // namespace Eigen

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Quat = Eigen::Quaterniond;

// hat: v -> [v]_x  (3x3 skew symmetric)
inline Mat3 hat(const Vec3 &v) {
    Mat3 m;
    m <<     0.0, -v.z(),  v.y(),
          v.z(),    0.0, -v.x(),
         -v.y(),  v.x(),    0.0;
    return m;
}

// vee: [v]_x -> v (optional)
// inline Vec3 vee(const Mat3 &M) { return Vec3(M(2,1), M(0,2), M(1,0)); }

// Exp map: so(3) (rotation vector) -> quaternion (SO(3) element)
// omega is rotation vector (axis * angle), units: radians. theta = |omega|.
inline Quat expmap_so3(const Vec3 &omega) {
    double theta = omega.norm();
    if (theta < 1e-12) {
        // small-angle approx: q ~= [1, omega/2]
        Vec3 half = 0.5 * omega;
        return Quat(1.0, half.x(), half.y(), half.z()).normalized();
    } else {
        Vec3 axis = omega / theta;
        double half_theta = 0.5 * theta;
        double w = std::cos(half_theta);
        double s = std::sin(half_theta);
        return Quat(w, axis.x() * s, axis.y() * s, axis.z() * s);
    }
}

// Right Jacobian Jr(omega) of SO(3).
// Uses standard closed-form:
//   A = (1 - cosθ) / θ^2
//   B = (θ - sinθ) / θ^3
// Jr = I - A * hat(omega) + B * hat(omega)^2
inline Mat3 rightJacobianSO3(const Vec3 &omega) {
    double theta = omega.norm();
    Mat3 I = Mat3::Identity();

    if (theta < 1e-8) {
        // Taylor expansion: Jr ≈ I - 0.5 * K + (1/12) * K^2
        Mat3 K = hat(omega);
        return I - 0.5 * K + (1.0/12.0) * (K * K);
    } else {
        Mat3 K = hat(omega);
        double theta2 = theta * theta;
        double A = (1.0 - std::cos(theta)) / theta2;
        double B = (theta - std::sin(theta)) / (theta2 * theta); // = (θ - sinθ)/θ^3
        return I - A * K + B * (K * K);
    }
}