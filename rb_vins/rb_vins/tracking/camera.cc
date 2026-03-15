/*
 * IC-GVINS: A Robust, Real-time, INS-Centric GNSS-Visual-Inertial Navigation System
 *
 * Copyright (C) 2022 i2Nav Group, Wuhan University
 *
 *     Author : Hailiang Tang
 *    Contact : thl@whu.edu.cn
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "tracking/camera.h"

Camera::Camera(Mat intrinsic, Mat distortion, const cv::Size &size)
    : distortion_(std::move(distortion))
    , intrinsic_(std::move(intrinsic)) {

    fx_   = intrinsic_.at<double>(0, 0);
    skew_ = intrinsic_.at<double>(0, 1);
    cx_   = intrinsic_.at<double>(0, 2);
    fy_   = intrinsic_.at<double>(1, 1);
    cy_   = intrinsic_.at<double>(1, 2);

    k1_ = distortion_.at<double>(0);
    k2_ = distortion_.at<double>(1);
    p1_ = distortion_.at<double>(2);
    p2_ = distortion_.at<double>(3);
    k3_ = distortion_.at<double>(4);

    width_  = size.width;
    height_ = size.height;

    // 相机畸变矫正初始化
    initUndistortRectifyMap(intrinsic_, distortion_, Mat(), intrinsic_, size, CV_16SC2, undissrc_, undisdst_);
}

Camera::Ptr Camera::createCamera(const std::vector<double> &intrinsic, const std::vector<double> &distortion,
                                 const std::vector<int> &size) {
    // Intrinsic matrix
    Mat intrinsic_mat;
    if (intrinsic.size() == 4) {
        intrinsic_mat =
            (cv::Mat_<double>(3, 3) << intrinsic[0], 0, intrinsic[2], 0, intrinsic[1], intrinsic[3], 0, 0, 1);
    } else if (intrinsic.size() == 5) {
        intrinsic_mat = (cv::Mat_<double>(3, 3) << intrinsic[0], intrinsic[4], intrinsic[2], 0, intrinsic[1],
                         intrinsic[3], 0, 0, 1);
    }

    // Distortion parameters
    Mat distortion_mat;
    if (distortion.size() == 4) {
        distortion_mat = (cv::Mat_<double>(5, 1) << distortion[0], distortion[1], distortion[2], distortion[3], 0.0);
    } else if (distortion.size() == 5) {
        distortion_mat =
            (cv::Mat_<double>(5, 1) << distortion[0], distortion[1], distortion[2], distortion[3], distortion[4]);
    }

    return std::make_shared<Camera>(intrinsic_mat, distortion_mat, cv::Size(size[0], size[1]));
}

void Camera::undistortPoints(std::vector<cv::Point2f> &pts) {
    cv::undistortPoints(pts, pts, intrinsic_, distortion_, Mat(), intrinsic_);
}

cv::Point2f Camera::undistortPointManual(const cv::Point2f &pt) {
    // 提取内参
    double fx = fx_;
    double fy = fy_;
    double cx = cx_;
    double cy = cy_;
    
    // 将像素点转换到归一化坐标系
    double x = (pt.x - cx) / fx;
    double y = (pt.y - cy) / fy;
    
    double x_corr, y_corr;
    if (0) {
        // 获取畸变系数，假设 distortion_ 的排列为 [k1, k2, p1, p2, k3]
        double k1 = k1_;
        double k2 = k2_;
        double p1 = p1_;
        double p2 = p2_;
        double k3 = k3_;
        
        // 计算径向畸变因子
        double r2 = x*x + y*y;
        double radial = 1 - k1*r2 - k2*r2*r2 - k3*r2*r2*r2;
        
        // 应用畸变校正
        x_corr = x * radial - 2*p1*x*y - p2*(r2 + 2*x*x);
        y_corr = y * radial - p1*(r2 + 2*y*y) - 2*p2*x*y;
    } else {
        // Recursive distortion model
        int n = 8;
        Eigen::Vector2d d_u;
        distortionstep(Eigen::Vector2d(x, y), d_u);
        // Approximate value
        x_corr = x - d_u(0);
        y_corr = y - d_u(1);

        for (int i = 1; i < n; ++i)
        {
            distortionstep(Eigen::Vector2d(x_corr, y_corr), d_u);
            x_corr = x - d_u(0);
            y_corr = y - d_u(1);
        }
    }
    
    // 将归一化坐标转换回像素坐标
    return cv::Point2f(static_cast<float>(x_corr * fx + cx),
                       static_cast<float>(y_corr * fy + cy));
}

void Camera::distortPoints(std::vector<cv::Point2f> &pts) const {
    for (auto &pt : pts) {
        auto pc   = pixel2cam(pt);
        double x  = pc.x();
        double y  = pc.y();
        double r2 = x * x + y * y;
        double rr = (1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2);

        pc.x() = x * rr + 2 * p1_ * x * y + p2_ * (r2 + 2 * x * x);
        pc.y() = y * rr + p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y;

        pt = cam2pixel(pc);
    }
}

void Camera::distortPoint(cv::Point2f &pp) const {
    auto pc   = pixel2cam(pp);
    double x  = pc.x();
    double y  = pc.y();
    double r2 = x * x + y * y;
    double rr = (1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2);

    pc.x() = x * rr + 2 * p1_ * x * y + p2_ * (r2 + 2 * x * x);
    pc.y() = y * rr + p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y;

    pp = cam2pixel(pc);
}

cv::Point2f Camera::distortCameraPoint(const Vector3d &pc) const {
    Vector3d pc1;

    double x  = pc.x() / pc.z();
    double y  = pc.y() / pc.z();
    double r2 = x * x + y * y;
    double rr = (1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2);

    pc1.x() = static_cast<float>(x * rr + 2 * p1_ * x * y + p2_ * (r2 + 2 * x * x));
    pc1.y() = static_cast<float>(y * rr + p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y);
    pc1.z() = 1.0;

    return cam2pixel(pc1);
}

void Camera::undistortImage(const Mat &src, Mat &dst) {
    cv::remap(src, dst, undissrc_, undisdst_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
}

Vector3d Camera::pixel2cam(const cv::Point2f &pixel) const {
    double y = (pixel.y - cy_) / fy_;
    double x = (pixel.x - cx_ - skew_ * y) / fx_;
    return {x, y, 1.0};
}

cv::Point2f Camera::cam2pixel(const Vector3d &cam) const {
    return cv::Point2f((fx_ * cam[0] + skew_ * cam[1]) / cam[2] + cx_, fy_ * cam[1] / cam[2] + cy_);
}

Vector3d Camera::pixel2unitcam(const cv::Point2f &pixel) const {
    return pixel2cam(pixel).normalized();
}

Vector3d Camera::pixel2world(const cv::Point2f &pixel, const Pose &pose) const {
    return cam2world(pixel2cam(pixel), pose);
}

cv::Point2f Camera::world2pixel(const Vector3d &world, const Pose &pose) const {
    return cam2pixel(world2cam(world, pose));
}

Vector3d Camera::world2cam(const Vector3d &world, const Pose &pose) {
    return pose.R.transpose() * (world - pose.t);
}

Vector3d Camera::cam2world(const Vector3d &cam, const Pose &pose) {
    return pose.R * cam + pose.t;
}

Vector2d Camera::reprojectionError(const Pose &pose, const Vector3d &pw, const cv::Point2f &pp) const {
    cv::Point2f ppp = world2pixel(pw, pose);

    return {ppp.x - pp.x, ppp.y - pp.y};
}

void Camera::distortionstep(const Vector2d &p_u, Vector2d &d_u) {
    double k1 = k1_;
    double k2 = k2_;
    double p1 = p1_;
    double p2 = p2_;
    double k3 = k3_;

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u + k3 * rho2_u * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

Pose Camera::posei2j(const Pose &pose_i, const Pose &pose_j) const {
    Pose pi2j;
    pi2j.R = pose_j.R.transpose() * pose_i.R;
    pi2j.t = pose_j.R.transpose() * (pose_i.t - pose_j.t);
    return pi2j;
}

Pose Camera::posei2k(const Pose &pi2j, const Pose &pj2k) const {
    Pose pi2k;
    pi2k.R = pj2k.R * pi2j.R;
    pi2k.t = pj2k.R * pi2j.t + pj2k.t;
    return pi2k;
}

Pose Camera::posej2w(const Pose &posei2w, const Pose &posei2j) const {
    Pose posej2w;
    posej2w.R = posei2w.R * posei2j.R.transpose();
    posej2w.t = posei2w.t - posei2w.R * posei2j.R.transpose() * posei2j.t;
    return posej2w;
}

Pose Camera::predictNextPose(const Pose& pose1, const Pose& pose2) const {
    Pose p21 = posei2j(pose1, pose2);

    Pose next;
    next.R = pose2.R * p21.R;
    next.t = pose2.R * p21.t + pose2.t;
    return next;
}

// Pose Camera::posej(const Pose &pji, const Pose &pi) const {
//     Pose pj;
//     pj.R = pji.R * pi.R;
//     pj.t = pji.R * pi.t + pji.t;
//     return pj;
// }

void Camera::computePoseError(const Pose &gt, const Pose &est, double &rot_deg, double &trans_error) {
    Eigen::Matrix3d R_err = est.R * gt.R.transpose();
    Eigen::AngleAxisd aa(R_err);
    rot_deg = aa.angle() * 180.0 / M_PI;
    trans_error = (gt.t - est.t).norm();
}

Pose Camera::computeIMUPoseFromCamera(const Pose &T_w_c, const Pose &T_b_c) {
    Pose T_w_b;

    // Step 1: compute inverse of T_b_c -> T_c_b
    Eigen::Matrix3d R_cb = T_b_c.R.transpose();
    Eigen::Vector3d t_cb = -R_cb * T_b_c.t;

    // Step 2: T_w_b = T_w_c * T_c_b
    T_w_b.R = T_w_c.R * R_cb;
    T_w_b.t = T_w_c.R * t_cb + T_w_c.t;

    return T_w_b;
}