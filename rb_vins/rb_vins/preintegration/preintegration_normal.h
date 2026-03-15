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

#ifndef PREINTEGRATION_NORMAL_H
#define PREINTEGRATION_NORMAL_H

#include "preintegration/preintegration_base.h"

class PreintegrationNormal : public PreintegrationBase {

public:
    PreintegrationNormal(std::shared_ptr<IntegrationParameters> parameters, const IMU &imu0, IntegrationState state);

    Eigen::MatrixXd evaluate(const IntegrationState &state0, const IntegrationState &state1,
                             double *residuals) override;

    Eigen::MatrixXd residualJacobianPose0(const IntegrationState &state0, const IntegrationState &state1,
                                          double *jacobian) override;
    Eigen::MatrixXd residualJacobianPose1(const IntegrationState &state0, const IntegrationState &state1,
                                          double *jacobian) override;
    Eigen::MatrixXd residualJacobianMix0(const IntegrationState &state0, const IntegrationState &state1,
                                         double *jacobian) override;
    Eigen::MatrixXd residualJacobianMix1(const IntegrationState &state0, const IntegrationState &state1,
                                         double *jacobian) override;
    int numResiduals() override;
    int numMixParametersBlocks() override;
    vector<int> numBlocksParameters() override;

    static IntegrationStateData stateToData(const IntegrationState &state);
    static IntegrationState stateFromData(const IntegrationStateData &data);
    void constructState(const double *const *parameters, IntegrationState &state0, IntegrationState &state1) override;

    //tmp change
    Eigen::MatrixXd sqrt_information() const {
        Eigen::MatrixXd sqrt_information =
        Eigen::LLT<Eigen::Matrix<double, NUM_STATE, NUM_STATE>>(covariance_.inverse()).matrixL().transpose();
        return sqrt_information;
    }

    Eigen::MatrixXd evaluate_raw(const IntegrationState &state0, const IntegrationState &state1) {
        // sqrt_information_ =
        //     Eigen::LLT<Eigen::Matrix<double, NUM_STATE, NUM_STATE>>(covariance_.inverse()).matrixL().transpose();

        Eigen::Matrix<double, NUM_STATE, 1> residual;

        Matrix3d dp_dbg = jacobian_.block<3, 3>(0, 9);
        Matrix3d dp_dba = jacobian_.block<3, 3>(0, 12);
        Matrix3d dv_dbg = jacobian_.block<3, 3>(3, 9);
        Matrix3d dv_dba = jacobian_.block<3, 3>(3, 12);
        Matrix3d dq_dbg = jacobian_.block<3, 3>(6, 9);

        // 零偏误差
        Vector3d dbg = state0.bg - delta_state_.bg;
        Vector3d dba = state0.ba - delta_state_.ba;

        // 积分校正
        corrected_p_ = delta_state_.p + dp_dba * dba + dp_dbg * dbg;
        corrected_v_ = delta_state_.v + dv_dba * dba + dv_dbg * dbg;
        corrected_q_ = delta_state_.q * Rotation::rotvec2quaternion(dq_dbg * dbg);

        // Residuals
        residual.block<3, 1>(0, 0) = state0.q.inverse() * (state1.p - state0.p - state0.v * delta_time_ -
                                                        0.5 * gravity_ * delta_time_ * delta_time_) - corrected_p_;
        residual.block<3, 1>(3, 0)  = state0.q.inverse() * (state1.v - state0.v - gravity_ * delta_time_) - corrected_v_;
        residual.block<3, 1>(6, 0)  = 2 * (corrected_q_.inverse() * state0.q.inverse() * state1.q).vec();
        residual.block<3, 1>(9, 0)  = state1.bg - state0.bg;
        residual.block<3, 1>(12, 0) = state1.ba - state0.ba;

        // residual = sqrt_information_ * residual;
        return residual;

    }

    std::shared_ptr<PreintegrationBase> clone() const override {
        return std::make_shared<PreintegrationNormal>(*this); // 使用默认拷贝构造
    }  

protected:
    void integrationProcess(unsigned long index) override;
    void resetState(const IntegrationState &state) override;

    void updateJacobianAndCovariance(const IMU &imu_pre, const IMU &imu_cur) override;

private:
    void resetState(const IntegrationState &state, int num);
    void setNoiseMatrix();

public:
    static constexpr int NUM_MIX = 9;

private:
    static constexpr int NUM_STATE = 15;
    static constexpr int NUM_NOISE = 12;

// tmp change
// public:
//     void integrationAndUpdate(const IMU &imu_pre, const IMU &imu_cur);
//     Eigen::Matrix<double, 6, 6> noise_gyro_acce_;
};

#endif // PREINTEGRATION_NORMAL_H
