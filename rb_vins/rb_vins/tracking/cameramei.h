#ifndef GVINS_CAMERA_MEI_H
#define GVINS_CAMERA_MEI_H

#include "camera.h"
#include <opencv2/ccalib/omnidir.hpp>

class MEICamera : public Camera {
public:
    typedef std::shared_ptr<MEICamera> Ptr;

    MEICamera(cv::Mat intrinsic, cv::Mat distortion, double xi, const cv::Size &size);

    static MEICamera::Ptr createCamera(const std::vector<double> &intrinsic, 
                                       const std::vector<double> &distortion,
                                       double xi,
                                       const std::vector<int> &size);

    void undistortPoints(std::vector<cv::Point2f> &pts) override;
    cv::Point2f undistortPoint(const cv::Point2f &pt);
    cv::Point2f undistortPointManual(const cv::Point2f &pt) override;
    void undistortImage(const cv::Mat &src, cv::Mat &dst) override;

    void distortPoints(std::vector<cv::Point2f> &pts) const override;
    void distortPoint(cv::Point2f &pp) const override;
    cv::Point2f distortCameraPoint(const Eigen::Vector3d &pc) const override;

    Vector3d pixel2cam(const cv::Point2f &pixel) const override;
    cv::Point2f cam2pixel(const Vector3d &cam) const override;

    double xi() const { return xi_; }

    double focalLength() const {
        return 720;//(fx_ + fy_) * 0.5 //460
    }


private:
    double xi_;  // MEI 相机特有参数
};

#endif // GVINS_MEI_CAMERA_H
