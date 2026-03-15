#ifndef VISUALPRIOR_FACTOR_H
#define VISUALPRIOR_FACTOR_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Geometry>

#include "common/rotation.h"


struct VisualPriorFactor {
    VisualPriorFactor(const Eigen::Matrix3d& R_prior,
                      const Eigen::Vector3d& t_prior,
                      const Eigen::Matrix<double,6,6>& sqrt_info)
        : R_prior_(R_prior), t_prior_(t_prior), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // 旋转是四元数
        Eigen::Quaternion<T> q_est(pose[6], pose[3], pose[4], pose[5]);
        Eigen::Matrix<T,3,3> R_est = q_est.normalized().toRotationMatrix();
        Eigen::Matrix<T,3,1> t_est(pose[0], pose[1], pose[2]);

        // 当前估计的 T_ref_cur
        Eigen::Matrix<T,3,3> R_prior_T = R_prior_.cast<T>().transpose(); // prior.inverse()
        Eigen::Matrix<T,3,1> t_prior_T = -R_prior_T * t_prior_.cast<T>();

        Eigen::Matrix<T,3,3> R_err = R_prior_T * R_est;
        Eigen::Matrix<T,3,1> t_err = R_prior_T * t_est + t_prior_T;

        // log(R_err), t_err → ξ (6x1 残差)
        Eigen::Matrix<T,6,1> xi;

        // 使用 Eigen::AngleAxis 来计算 log(R)
        Eigen::AngleAxis<T> aa(R_err);
        Eigen::Matrix<T,3,1> omega = aa.angle() * aa.axis();
        T theta = omega.norm();
        Eigen::Matrix<T,3,3> Omega_hat;
        Omega_hat <<        T(0), -omega(2), omega(1),
                      omega(2),        T(0), -omega(0),
                     -omega(1), omega(0),        T(0);

        Eigen::Matrix<T,3,3> V_inv = Eigen::Matrix<T,3,3>::Identity();
        if (theta > T(1e-5)) {
            T half_theta = T(0.5) * theta;
            T cot_half_theta = T(1.0) / tan(half_theta);
            V_inv = Eigen::Matrix<T,3,3>::Identity()
                  - T(0.5) * Omega_hat
                  + (T(1.0) / (theta * theta)) * (T(1.0) - (theta * cot_half_theta / T(2.0))) * (Omega_hat * Omega_hat);
        }

        Eigen::Matrix<T,3,1> upsilon = V_inv * t_err;

        xi.template head<3>() = omega;
        xi.template tail<3>() = upsilon;

        // 残差 = sqrt_info * xi
        Eigen::Map<Eigen::Matrix<T,6,1>> res(residuals);
        res = sqrt_info_.template cast<T>() * xi;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix3d& R_prior,
                                       const Eigen::Vector3d& t_prior,
                                       const Eigen::Matrix<double,6,6>& sqrt_info) {
        return new ceres::AutoDiffCostFunction<VisualPriorFactor, 6, 7>(
            new VisualPriorFactor(R_prior, t_prior, sqrt_info));
    }

    Eigen::Matrix3d R_prior_;
    Eigen::Vector3d t_prior_;
    Eigen::Matrix<double,6,6> sqrt_info_;
};

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

#endif // VISUALPRIOR_FACTOR_H
