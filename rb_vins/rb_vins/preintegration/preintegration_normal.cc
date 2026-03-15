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

#include "preintegration/preintegration_normal.h"

#include "common/logging.h"

#include "../common/eigentypes.h"

PreintegrationNormal::PreintegrationNormal(std::shared_ptr<IntegrationParameters> parameters, const IMU &imu0,
                                           IntegrationState state)
    : PreintegrationBase(std::move(parameters), imu0, std::move(state)) {

    // Reset state
    resetState(state, NUM_STATE);

    // Set initial noise matrix
    setNoiseMatrix();
}

Eigen::MatrixXd PreintegrationNormal::evaluate(const IntegrationState &state0, const IntegrationState &state1,
                                               double *residuals) {
    sqrt_information_ =
        Eigen::LLT<Eigen::Matrix<double, NUM_STATE, NUM_STATE>>(covariance_.inverse()).matrixL().transpose();

    Eigen::Map<Eigen::Matrix<double, NUM_STATE, 1>> residual(residuals);

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
                                                       0.5 * gravity_ * delta_time_ * delta_time_) -
                                 corrected_p_;
    residual.block<3, 1>(3, 0)  = state0.q.inverse() * (state1.v - state0.v - gravity_ * delta_time_) - corrected_v_;
    residual.block<3, 1>(6, 0)  = 2 * (corrected_q_.inverse() * state0.q.inverse() * state1.q).vec();
    residual.block<3, 1>(9, 0)  = state1.bg - state0.bg;
    residual.block<3, 1>(12, 0) = state1.ba - state0.ba;

    residual = sqrt_information_ * residual;
    return residual;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianPose0(const IntegrationState &state0,
                                                            const IntegrationState &state1, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_POSE, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    jaco.block(0, 0, 3, 3) = -state0.q.inverse().toRotationMatrix();
    jaco.block(0, 3, 3, 3) =
        Rotation::skewSymmetric(state0.q.inverse() * (state1.p - state0.p - state0.v * delta_time_ -
                                                      0.5 * gravity_ * delta_time_ * delta_time_));
    jaco.block(3, 3, 3, 3) =
        Rotation::skewSymmetric(state0.q.inverse() * (state1.v - state0.v - gravity_ * delta_time_));
    jaco.block(6, 3, 3, 3) =
        -(Rotation::quaternionleft(state1.q.inverse() * state0.q) * Rotation::quaternionright(corrected_q_))
             .bottomRightCorner<3, 3>();

    jaco = sqrt_information_ * jaco;
    return jaco;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianPose1(const IntegrationState &state0,
                                                            const IntegrationState &state1, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_POSE, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    jaco.block(0, 0, 3, 3) = state0.q.inverse().toRotationMatrix();
    jaco.block(6, 3, 3, 3) =
        Rotation::quaternionleft(corrected_q_.inverse() * state0.q.inverse() * state1.q).bottomRightCorner<3, 3>();

    jaco = sqrt_information_ * jaco;
    return jaco;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianMix0(const IntegrationState &state0,
                                                           const IntegrationState &state1, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_MIX, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(0, 9);
    Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(0, 12);
    Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(3, 9);
    Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(3, 12);
    Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(6, 9);

    jaco.block(0, 0, 3, 3) = -state0.q.inverse().toRotationMatrix() * delta_time_;
    jaco.block(0, 3, 3, 3) = -dp_dbg;
    jaco.block(0, 6, 3, 3) = -dp_dba;
    jaco.block(3, 0, 3, 3) = -state0.q.inverse().toRotationMatrix();
    jaco.block(3, 3, 3, 3) = -dv_dbg;
    jaco.block(3, 6, 3, 3) = -dv_dba;
    jaco.block(6, 3, 3, 3) =
        -Rotation::quaternionleft(state1.q.inverse() * state0.q * delta_state_.q).bottomRightCorner<3, 3>() * dq_dbg;
    jaco.block(9, 3, 3, 3)  = -Eigen::Matrix3d::Identity();
    jaco.block(12, 6, 3, 3) = -Eigen::Matrix3d::Identity();

    jaco = sqrt_information_ * jaco;
    return jaco;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianMix1(const IntegrationState &state0,
                                                           const IntegrationState &state1, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_MIX, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    jaco.block(3, 0, 3, 3)  = state0.q.inverse().toRotationMatrix();
    jaco.block(9, 3, 3, 3)  = Eigen::Matrix3d::Identity();
    jaco.block(12, 6, 3, 3) = Eigen::Matrix3d::Identity();

    jaco = sqrt_information_ * jaco;
    return jaco;
}

int PreintegrationNormal::numResiduals() {
    return NUM_STATE;
}

vector<int> PreintegrationNormal::numBlocksParameters() {
    return std::vector<int>{NUM_POSE, NUM_MIX, NUM_POSE, NUM_MIX};
}

IntegrationStateData PreintegrationNormal::stateToData(const IntegrationState &state) {
    IntegrationStateData data;
    PreintegrationBase::stateToData(state, data);
    return data;
}

IntegrationState PreintegrationNormal::stateFromData(const IntegrationStateData &data) {
    IntegrationState state;
    PreintegrationBase::stateFromData(data, state);
    return state;
}

void PreintegrationNormal::constructState(const double *const *parameters, IntegrationState &state0,
                                          IntegrationState &state1) {
    state0 = IntegrationState{
        .p  = {parameters[0][0], parameters[0][1], parameters[0][2]},
        .q  = {parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]},
        .v  = {parameters[1][0], parameters[1][1], parameters[1][2]},
        .bg = {parameters[1][3], parameters[1][4], parameters[1][5]},
        .ba = {parameters[1][6], parameters[1][7], parameters[1][8]},
    };

    state1 = IntegrationState{
        .p  = {parameters[2][0], parameters[2][1], parameters[2][2]},
        .q  = {parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]},
        .v  = {parameters[3][0], parameters[3][1], parameters[3][2]},
        .bg = {parameters[3][3], parameters[3][4], parameters[3][5]},
        .ba = {parameters[3][6], parameters[3][7], parameters[3][8]},
    };
}

void PreintegrationNormal::integrationProcess(unsigned long index) {
    IMU imu_pre = compensationBias(imu_buffer_[index - 1]);
    IMU imu_cur = compensationBias(imu_buffer_[index]);

    // 连续状态积分和预积分
    integration(imu_pre, imu_cur);

    // 更新系统状态雅克比和协方差矩阵
    updateJacobianAndCovariance(imu_pre, imu_cur);

    // integrationAndUpdate(imu_pre, imu_cur);
}

void PreintegrationNormal::resetState(const IntegrationState &state) {
    resetState(state, NUM_STATE);
}

void PreintegrationNormal::updateJacobianAndCovariance(const IMU &imu_pre, const IMU &imu_cur) {
    // dp, dv, dq, dbg, dba

    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(NUM_STATE, NUM_STATE);

    double dt = imu_cur.dt;

    // jacobian

    // phi = I + F * dt
    phi.block<3, 3>(0, 0)   = Matrix3d::Identity();
    phi.block<3, 3>(0, 3)   = Matrix3d::Identity() * dt;
    phi.block<3, 3>(3, 3)   = Matrix3d::Identity();
    phi.block<3, 3>(3, 6)   = -delta_state_.q.toRotationMatrix() * Rotation::skewSymmetric(imu_cur.dvel);
    phi.block<3, 3>(3, 12)  = -delta_state_.q.toRotationMatrix() * dt;
    phi.block<3, 3>(6, 6)   = Matrix3d::Identity() - Rotation::skewSymmetric(imu_cur.dtheta);
    phi.block<3, 3>(6, 9)   = -Matrix3d::Identity() * dt;
    phi.block<3, 3>(9, 9)   = Matrix3d::Identity() * (1 - dt / parameters_->corr_time);
    phi.block<3, 3>(12, 12) = Matrix3d::Identity() * (1 - dt / parameters_->corr_time);

    // // tmp change
    // // phi = I + F * dt
    // phi.block<3, 3>(0, 0)   = Matrix3d::Identity();
    // phi.block<3, 3>(0, 3)   = Matrix3d::Identity() * dt;
    // phi.block<3, 3>(0, 6)   = -0.5 * delta_state_.q.toRotationMatrix() * Rotation::skewSymmetric(imu_cur.dvel) * dt;
    // phi.block<3, 3>(0, 12)  = -0.5 * delta_state_.q.toRotationMatrix() * dt * dt;
    // phi.block<3, 3>(3, 3)   = Matrix3d::Identity();
    // phi.block<3, 3>(3, 6)   = -delta_state_.q.toRotationMatrix() * Rotation::skewSymmetric(imu_cur.dvel);
    // phi.block<3, 3>(3, 12)  = -delta_state_.q.toRotationMatrix() * dt;
    // phi.block<3, 3>(6, 6)   = Matrix3d::Identity() - Rotation::skewSymmetric(imu_cur.dtheta);
    // phi.block<3, 3>(6, 9)   = -Matrix3d::Identity() * dt;
    // phi.block<3, 3>(9, 9)   = Matrix3d::Identity() * (1 - dt / parameters_->corr_time);
    // phi.block<3, 3>(12, 12) = Matrix3d::Identity() * (1 - dt / parameters_->corr_time);

    jacobian_ = phi * jacobian_;

    // covariance

    Eigen::MatrixXd gt = Eigen::MatrixXd::Zero(NUM_STATE, NUM_NOISE);

    gt.block<3, 3>(3, 3)  = delta_state_.q.toRotationMatrix();
    gt.block<3, 3>(6, 0)  = Matrix3d::Identity();
    gt.block<3, 3>(9, 6)  = Matrix3d::Identity();
    gt.block<3, 3>(12, 9) = Matrix3d::Identity();

    Eigen::MatrixXd Qk =
        0.5 * dt * (phi * gt * noise_ * gt.transpose() + gt * noise_ * gt.transpose() * phi.transpose());
    covariance_ = phi * covariance_ * phi.transpose() + Qk;
}

// tmp change
// void PreintegrationNormal::integrationAndUpdate(const IMU &imu_pre, const IMU &imu_cur) {
//     // 区间时间累积
//     double dt = imu_cur.dt;
//     delta_time_ += dt;

//     end_time_           = imu_cur.time;
//     current_state_.time = imu_cur.time;

//     // 连续状态积分, 先位置速度再姿态

//     // 位置速度
//     Vector3d dvfb = imu_cur.dvel + 0.5 * imu_cur.dtheta.cross(imu_cur.dvel) +
//                     1.0 / 12.0 * (imu_pre.dtheta.cross(imu_cur.dvel) + imu_pre.dvel.cross(imu_cur.dtheta));
//     Vector3d dvel = current_state_.q.toRotationMatrix() * dvfb + gravity_ * dt;

//     current_state_.p += dt * current_state_.v + 0.5 * dt * dvel;
//     current_state_.v += dvel;

//     // 姿态
//     Vector3d dtheta = imu_cur.dtheta + 1.0 / 12.0 * imu_pre.dtheta.cross(imu_cur.dtheta);
//     current_state_.q *= Rotation::rotvec2quaternion(dtheta);
//     current_state_.q.normalize();

//     // // 预积分
//     // dvel = delta_state_.q.toRotationMatrix() * dvfb;
//     // delta_state_.p += dt * delta_state_.v + 0.5 * dt * dvel;
//     // delta_state_.v += dvel;

//     // // 姿态
//     // delta_state_.q *= Rotation::rotvec2quaternion(dtheta);
//     // delta_state_.q.normalize();

//     // 去掉零偏的测量
//     Vector3d gyr = imu_cur.dtheta / dt;  // 陀螺
//     Vector3d acc = imu_cur.dvel / dt;  // 加计

//     // 更新dv, dp, 见(4.13), (4.16)
//     delta_state_.p = delta_state_.p + delta_state_.v * dt + 0.5f * delta_state_.q.toRotationMatrix() * imu_cur.dvel * dt;
//     delta_state_.v = delta_state_.v + delta_state_.q.toRotationMatrix() * imu_cur.dvel;

//     // dR先不更新，因为A, B阵还需要现在的dR

//     // 运动方程雅可比矩阵系数，A,B阵，见(4.29)
//     // 另外两项在后面
//     Eigen::Matrix<double, 9, 9> A;
//     A.setIdentity();
//     Eigen::Matrix<double, 9, 6> B;
//     B.setZero();

//     // Eigen::Matrix3d acc_hat = hat(acc);
//     Eigen::Matrix3d dvel_hat = hat(imu_cur.dvel);
//     double dt2 = dt * dt;

//     Eigen::Matrix3d dP_dbg_ = jacobian_.block<3, 3>(0, 9);
//     Eigen::Matrix3d dP_dba_ = jacobian_.block<3, 3>(0, 12);
//     Eigen::Matrix3d dV_dbg_ = jacobian_.block<3, 3>(3, 9);
//     Eigen::Matrix3d dV_dba_ = jacobian_.block<3, 3>(3, 12);
//     Eigen::Matrix3d dR_dbg_ = jacobian_.block<3, 3>(6, 9);

//     // NOTE A, B左上角块与公式稍有不同
//     A.block<3, 3>(3, 0) = -delta_state_.q.toRotationMatrix() * dvel_hat;
//     A.block<3, 3>(6, 0) = -0.5f * delta_state_.q.toRotationMatrix() * dvel_hat * dt;
//     A.block<3, 3>(6, 3) = dt * Eigen::Matrix3d::Identity();

//     B.block<3, 3>(3, 3) = delta_state_.q.toRotationMatrix() * dt;
//     B.block<3, 3>(6, 3) = 0.5f * delta_state_.q.toRotationMatrix() * dt2;

//     // 更新各雅可比，见式(4.39)
//     dP_dba_ = dP_dba_ + dV_dba_ * dt - 0.5f * delta_state_.q.toRotationMatrix() * dt2;                      // (4.39d)
//     dP_dbg_ = dP_dbg_ + dV_dbg_ * dt - 0.5f * delta_state_.q.toRotationMatrix() * dt * dvel_hat * dR_dbg_;  // (4.39e)
//     dV_dba_ = dV_dba_ - delta_state_.q.toRotationMatrix() * dt;                                             // (4.39b)
//     dV_dbg_ = dV_dbg_ - delta_state_.q.toRotationMatrix() * dvel_hat * dR_dbg_;                         // (4.39c)

//     // 旋转部分
//     Eigen::Matrix3d rightJ = rightJacobianSO3(imu_cur.dtheta);  // 右雅可比
//     Eigen::Quaterniond deltaR = expmap_so3(imu_cur.dtheta);   // exp后
//     delta_state_.q = delta_state_.q * deltaR;             // (4.9)
//     delta_state_.q.normalized();

//     A.block<3, 3>(0, 0) = deltaR.toRotationMatrix().transpose();
//     B.block<3, 3>(0, 0) = rightJ * dt;

//     // 更新噪声项
//     covariance_ = A * covariance_ * A.transpose() + B * noise_gyro_acce_ * B.transpose();

//     // 更新dR_dbg
//     dR_dbg_ = deltaR.toRotationMatrix().transpose() * dR_dbg_ - rightJ * dt;  // (4.39a)

//     jacobian_.block<3, 3>(0, 9) = dP_dbg_;
//     jacobian_.block<3, 3>(0, 12) = dP_dba_;
//     jacobian_.block<3, 3>(3, 9) = dV_dbg_;
//     jacobian_.block<3, 3>(3, 12) = dV_dba_;
//     jacobian_.block<3, 3>(6, 9) = dR_dbg_;

// }

void PreintegrationNormal::resetState(const IntegrationState &state, int num) {
    delta_time_ = 0;
    delta_state_.p.setZero();
    delta_state_.q.setIdentity();
    delta_state_.v.setZero();
    delta_state_.bg = state.bg;
    delta_state_.ba = state.ba;

    jacobian_.setIdentity(num, num);
    covariance_.setZero(num, num);
}

void PreintegrationNormal::setNoiseMatrix() {
    noise_.setIdentity(NUM_NOISE, NUM_NOISE);
    noise_.block<3, 3>(0, 0) *= parameters_->gyr_arw * parameters_->gyr_arw; // nw
    noise_.block<3, 3>(3, 3) *= parameters_->acc_vrw * parameters_->acc_vrw; // na
    noise_.block<3, 3>(6, 6) *=
        2 * parameters_->gyr_bias_std * parameters_->gyr_bias_std / parameters_->corr_time; // nbg
    noise_.block<3, 3>(9, 9) *=
        2 * parameters_->acc_bias_std * parameters_->acc_bias_std / parameters_->corr_time; // nba

    // // tmp change
    // // ==== 6x6测量噪声矩阵(离散化后)，用于预积分B*Q*B^T ====
    // noise_gyro_acce_.setZero();
    // double dt = 1.0 / 200.0;

    // // 离散时间方差 = 连续时间噪声密度² / dt
    // double gyr_var = parameters_->gyr_arw * parameters_->gyr_arw / dt;
    // double acc_var = parameters_->acc_vrw * parameters_->acc_vrw / dt;

    // noise_gyro_acce_.block<3, 3>(0, 0) = gyr_var * Eigen::Matrix3d::Identity();
    // noise_gyro_acce_.block<3, 3>(3, 3) = acc_var * Eigen::Matrix3d::Identity();
}

int PreintegrationNormal::numMixParametersBlocks() {
    return NUM_MIX;
}
