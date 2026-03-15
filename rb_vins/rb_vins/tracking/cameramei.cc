#include "cameramei.h"

MEICamera::MEICamera(cv::Mat intrinsic, cv::Mat distortion, double xi, const cv::Size &size)
    : Camera(intrinsic, distortion, size), xi_(xi) {}

MEICamera::Ptr MEICamera::createCamera(const std::vector<double> &intrinsic, 
                                       const std::vector<double> &distortion, 
                                       double xi, 
                                       const std::vector<int> &size) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << intrinsic[0], 0, intrinsic[2],
                                           0, intrinsic[1], intrinsic[3],
                                           0, 0, 1);
    cv::Mat D = (cv::Mat_<double>(4, 1) << distortion[0], distortion[1], distortion[2], distortion[3]);
    return std::make_shared<MEICamera>(K, D, xi, cv::Size(size[0], size[1]));
}

// **MEI 相机去畸变特征点**
void MEICamera::undistortPoints(std::vector<cv::Point2f> &pts) {
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        cv::Point2f b = undistortPointManual(pts[i]);
        pts[i] = b;
    }
}

// **MEI 相机去畸变单个点**
cv::Point2f MEICamera::undistortPoint(const cv::Point2f &pt) {
    std::vector<cv::Point2f> pts = {pt};
    undistortPoints(pts);
    return pts[0];
}

cv::Point2f MEICamera::undistortPointManual(const cv::Point2f &pt) {
    // 1. 计算归一化坐标
    double x = (pt.x - cx_) / fx_;
    double y = (pt.y - cy_) / fy_;

    double x_corr, y_corr;
    if (0) {
       // 计算径向畸变因子
        double r2 = x*x + y*y;
        double radial = 1 - k1_*r2 - k2_*r2*r2 - k3_*r2*r2*r2;
        
        // 应用畸变校正
        x_corr = x * radial - 2*p1_*x*y - p2_*(r2 + 2*x*x);
        y_corr = y * radial - p1_*(r2 + 2*y*y) - 2*p2_*x*y;
    } else {
        x_corr = x;
        y_corr = y;
        // Recursive distortion model
        double r2, r4, r6, cdest, a1, a2, a3, deltaX, deltaY;
        for(int i=0; i<10; i++)
        {
            r2 =x_corr*x_corr + y_corr*y_corr;
            r4 = r2 * r2;
            r6 = r2 * r2 * r2;

            cdest = k1_*r2 + k2_*r4 + k3_*r6;
            a1 = 2.f * x_corr * y_corr;
            a2 = r2 + 2 * x_corr * x_corr;
            a3 = r2 + 2 * y_corr * y_corr;

            deltaX = p1_ * a1 + p2_ * a2;
            deltaY = p1_ * a3 + p2_ * a1;

            x_corr = x - x_corr * cdest - deltaX;
            y_corr = y - y_corr * cdest - deltaY;
        }
    }

    double Xn, Yn;
    // 将归一化平面的点投影到归一化球坐标系中
    // double xi = xi_;
    // double lambda;
    // if (xi == 1.0)
    // {
    //     lambda = 2.0 / (x_corr * x_corr + y_corr * y_corr + 1.0);
    //     // P << lambda * mx_u, lambda * my_u, lambda - 1.0;
    //     Xn = (lambda * x_corr) / (lambda - 1.0);
    //     Yn = (lambda * y_corr) / (lambda - 1.0);
    // }
    // else
    // {
    //     lambda = (xi + sqrt(1.0 + (1.0 - xi * xi) * (x_corr * x_corr + y_corr * y_corr))) / (1.0 + x_corr * x_corr + y_corr * y_corr);
    //     // P << lambda * mx_u, lambda * my_u, lambda - xi;
    //     Xn = (lambda * x_corr) / (lambda - xi);
    //     Yn = (lambda * y_corr) / (lambda - xi);
    // }

    Xn = x_corr;
    Yn = y_corr;

    
    // 5. 转换回像素坐标
    double u_undist = fx_ * Xn + cx_;
    double v_undist = fy_ * Yn + cy_;

    return cv::Point2f(u_undist, v_undist);
}

// **MEI 相机去畸变整个图像**
void MEICamera::undistortImage(const cv::Mat &src, cv::Mat &dst) {
    cv::omnidir::undistortImage(src, dst, intrinsic_, distortion_, xi_, 
                                cv::omnidir::RECTIFY_PERSPECTIVE, intrinsic_, src.size());
}

// **MEI 相机添加畸变**
void MEICamera::distortPoints(std::vector<cv::Point2f> &pts) const {
    double xi = xi_;
    for (auto &pt : pts) {
        double y = (pt.y - cy_) / fy_;
        double x = (pt.x - cx_ - skew_ * y) / fx_;
        double r2 = x * x + y * y;
        double rr = (1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2);

        double Xd = x * rr + 2 * p1_ * x * y + p2_ * (r2 + 2 * x * x);
        double Yd = y * rr + p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y;

        pt.x = fx_ * Xd + skew_ * Yd + cx_;
        pt.y = fy_ * Yd + cy_;
    }
}

void MEICamera::distortPoint(cv::Point2f &pp) const {
    double xi = xi_;
    double y = (pp.y - cy_) / fy_;
    double x = (pp.x - cx_ - skew_ * y) / fx_;
    double r2 = x * x + y * y;
    double rr = (1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2);

    double Xd = x * rr + 2 * p1_ * x * y + p2_ * (r2 + 2 * x * x);
    double Yd = y * rr + p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y;

    pp.x = fx_ * Xd + skew_ * Yd + cx_;
    pp.y = fy_ * Yd + cy_;
}

cv::Point2f MEICamera::distortCameraPoint(const Eigen::Vector3d &pc) const {
    double xi = xi_;
    double Xd, Yd;
    // 1. 归一化相机坐标
    auto pc_n = pc.normalized();
    double x  = pc_n[0] / (pc_n[2] + xi);
    double y  = pc_n[1] / (pc_n[2] + xi);
        
    double r2 = x * x + y * y;
    double rr = (1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2);

    Xd = x * rr + 2 * p1_ * x * y + p2_ * (r2 + 2 * x * x);
    Yd = y * rr + p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y;

    // 3. 转换回像素坐标
    double u = fx_ * Xd + skew_ * Yd + cx_;
    double v = fy_ * Yd + cy_;

    return cv::Point2f(u, v);
}

Vector3d MEICamera::pixel2cam(const cv::Point2f &pixel) const {
    double Xn, Yn;
    double mx_u = (pixel.x - cx_) / fx_;
    double my_u = (pixel.y - cy_) / fy_;
    double xi = xi_;
    double lambda;
    if (xi == 1.0)
    {
        lambda = 2.0 / (mx_u * mx_u + my_u * my_u + 1.0);
        // P << lambda * mx_u, lambda * my_u, lambda - 1.0;
        Xn = (lambda * mx_u) / (lambda - 1.0);
        Yn = (lambda * my_u) / (lambda - 1.0);
    }
    else
    {
        lambda = (xi + sqrt(1.0 + (1.0 - xi * xi) * (mx_u * mx_u + my_u * my_u))) / (1.0 + mx_u * mx_u + my_u * my_u);
        // P << lambda * mx_u, lambda * my_u, lambda - xi;
        Xn = (lambda * mx_u) / (lambda - xi);
        Yn = (lambda * my_u) / (lambda - xi);
    }
    double y = Yn;
    double x = Xn;
    return {x, y, 1.0};
}

cv::Point2f MEICamera::cam2pixel(const Vector3d &cam) const {
    double Xn, Yn;
    double xi = xi_;
    auto cam_n = cam.normalized();
    Xn = cam_n[0] / (cam_n[2] + xi);
    Yn = cam_n[1] / (cam_n[2] + xi);
    double x = fx_ * Xn + skew_ * Yn + cx_;
    double y = fy_ * Yn + cy_;
    return cv::Point2f(x, y);
}