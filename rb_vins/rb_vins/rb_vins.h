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

#ifndef GVINS_GVINS_H
#define GVINS_GVINS_H

#include "common/angle.h"
#include "common/timecost.h"
#include "fileio/filesaver.h"
#include "tracking/drawer.h"
#include "tracking/tracking.h"
#include "tracking/cameramei.h"

#include "factors/marginalization_info.h"
#include "factors/reprojection_factor.h"
#include "factors/reprojection_factor_w.h"
#include "preintegration/preintegration.h"

#include "morefuncs/utility.h"
#include "morefuncs/polynomial.h"

#include <ceres/ceres.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <unordered_map>

struct ResidualInfo {
    double w;
    ceres::ResidualBlockId id;
    ceres::CostFunction* cost_function;

    ResidualInfo(double w_, ceres::ResidualBlockId id_, ceres::CostFunction* cost_function_)
    : w(w_), id(id_), cost_function(cost_function_) {}

};

// 超参数，可按需调整
const double alpha_val = 2.5;  // 控制平均误差影响
const double beta_val  = 1.5;  // 控制方差影响
const double gamma_val = 0.1;  // 次数平滑因子

// RPE 统计结构
struct RpeStats {
    double mean{0}, var{0};
    int count{0};
};

class GVINS {

public:
    enum GVINSState {
        GVINS_ERROR                 = -1,
        GVINS_INITIALIZING          = 0,
        GVINS_INITIALIZING_INS      = 1,
        GVINS_INITIALIZING_VIO      = 2,
        GVINS_TRACKING_INITIALIZING = 3,
        GVINS_TRACKING_NORMAL       = 4,
        GVINS_TRACKING_LOST         = 5,
    };

    typedef std::shared_ptr<GVINS> Ptr;
    typedef std::unique_lock<std::mutex> Lock;

    GVINS() = delete;
    explicit GVINS(const string &configfile, const string &outputpath, Drawer::Ptr drawer);

    bool addNewImu(const IMU &imu);
    bool addNewFrame(const Frame::Ptr &frame);

    void setFinished();

    bool isRunning() const {
        return !isfinished_;
    }

    GVINSState gvinsState() const {
        return gvinsstate_;
    }

private:
    void parametersStatistic();

    bool gvinsInitialization();
    bool gvinsInitializationOptimization();
    // bool gvinsInitOptimization();

    void addNewTimeNode(double time);
    // void addNewGnssTimeNode();
    // bool insertNewGnssTimeNode();
    void addNewKeyFrameTimeNode();
    bool removeUnusedTimeNode();
    void constructPrior(bool is_zero_velocity);

    void addStateParameters(ceres::Problem &problem);
    void addReprojectionParameters(ceres::Problem &problem);

    void addImuFactors(ceres::Problem &problem);
    void addImuFactors_loss(ceres::Problem &problem);
    // vector<std::pair<ceres::ResidualBlockId, GNSS *>> addGnssFactors(ceres::Problem &problem, bool isusekernel);
    vector<ceres::ResidualBlockId> addReprojectionFactors(ceres::Problem &problem, bool isusekernel);
    void doReintegration();

    void updateParametersFromOptimizer();

    int getStateDataIndex(double time);

    bool gvinsOptimization();
    bool gvinsMarginalization();
    bool gvinsOutlierCulling();
    bool gvinsRemoveAllSecondNewFrame();

    void gnssOutlierCullingByChi2(ceres::Problem &problem,
                                  vector<std::pair<ceres::ResidualBlockId, GNSS *>> &redisual_block);
    static int removeReprojectionFactorsByChi2(ceres::Problem &problem, vector<ceres::ResidualBlockId> &residual_ids,
                                               double chi2);

    // Processing thread
    void runFusion();
    void runTracking();
    void runOptimization();

    //add vio init
    void resetGyroBias(const double N);
    bool weightgyroBiasEstimator(const std::map<int, double> &int_frameid2_time_frameid, Eigen::Vector3d &biasg);
    void solve_ws(const std::map<int, double> &int_frameid2_time_frameid, std::vector<std::vector<double>> &ws);
    void solve_res(const std::vector<Eigen::Vector3d> &fis, const std::vector<Eigen::Vector3d> &fjs, const std::vector<double> &w,
                   std::shared_ptr<PreintegrationBase> imu1, Eigen::VectorXd &res);
    void select_base_views(const Eigen::aligned_map<double, Eigen::Matrix3d> &frame_rot, const Eigen::aligned_map<double, FeaturePerFrame> &track, 
                           const std::map<int, double> &int_frameid2_time_frameid, const std::map<double, int> &time_frameid2_int_frameid, 
                           double &lbase_view_id, double &rbase_view_id);
    void select_base_views(const Eigen::aligned_map<double, Eigen::Matrix3d> &frame_rot, const Eigen::aligned_unordered_map<double, std::weak_ptr<Feature>> &track, 
                           const std::map<int, double> &int_frameid2_time_frameid, const std::map<double, int> &time_frameid2_int_frameid, 
                           double &lbase_view_id, double &rbase_view_id);
    void build_LTL(const Eigen::aligned_map<double, Eigen::Matrix3d> &frame_rot, const std::map<int, double> &int_frameid2_time_frameid, 
                   const std::map<double, int> &time_frameid2_int_frameid, Eigen::MatrixXd &LTL, Eigen::MatrixXd &A_lr, std::vector<double> &all_weights);
    void build_LTL(const Eigen::aligned_map<double, Eigen::Matrix3d> &frame_rot, const std::map<int, double> &int_frameid2_time_frameid, 
                   const std::map<double, int> &time_frameid2_int_frameid, Eigen::MatrixXd &LTL, Eigen::MatrixXd &A_lr);
    bool solve_LTL(const Eigen::MatrixXd &LTL, Eigen::VectorXd &evectors);
    bool checkHessianStability(double condition_number);
    void identify_sign(const Eigen::MatrixXd &A_lr, Eigen::VectorXd &evectors);
    bool linearAlignment(const std::map<int, double> &int_frameid2_time_frameid, const std::map<double, int> &time_frameid2_int_frameid, 
                         std::vector<Eigen::Vector3d> &velocity, std::vector<Eigen::Vector3d> &position, std::vector<Eigen::Matrix3d> &rotation);
    bool gravityRefine(const Eigen::MatrixXd &M, const Eigen::VectorXd &m, double Q, double gravity_mag, Eigen::VectorXd &rhs);
    void slidewindow();
    bool gvinsOptimization_old();
    void addImuFactors_old(ceres::Problem &problem);
    void addReprojectionParameters_old(ceres::Problem &problem);
    vector<ResidualInfo> addReprojectionFactors_old(ceres::Problem &problem, bool isusekernel);
    static int removeReprojectionFactorsByChi2_old(ceres::Problem &problem, vector<ResidualInfo> &residual_ids,
                                                    double chi2);
    void gnc_update(ceres::Problem &problem, vector<ResidualInfo> &ResidualInfo_ids,
                       double mu, double noise_bound_sq, double &cost_sum, double &w_sum);
    void addVisualpriorFactor(ceres::Problem &problem);
    // void setStateParametersConstant(ceres::Problem &problem);
    // void setStateParametersVariable(ceres::Problem &problem);
    void fixUnstableInverseDepths(ceres::Problem& problem);
    bool gvinsIncrementalMarginalization();
    bool removeUnusedTimeNode_old();
    bool gvinsSubOptimization();
    void addStateParameters_old(ceres::Problem &problem);

    void updateParametersFromOptimizer_old();
    bool gvinsOptimization_new();

private:
    // 正常重力
    // Normal gravity
    const double NORMAL_GRAVITY = -9.81007;  //9.80

    // INS窗口内的最大数量, 对于200Hz, 保留5秒数据
    // Maximum INS data in the window
    const size_t MAXIMUM_INS_NUMBER = 1000; //1000

    // 动态航向初始的最小速度
    // Minimum velocity for GNSS/INS intializaiton
    const double MINMUM_ALIGN_VELOCITY = 0.5;

    // 允许的最小同步间隔
    // Minimum synchronization interval for GNSS
    const double MINMUM_SYNC_INTERVAL = 0.025;

    // 允许的最长预积分时间
    // Maximum length for IMU preintegration
    const double MAXIMUM_PREINTEGRATION_LENGTH = 10.0;

    // 先验标准差
    // The prior STD for IMU biases
    const double GYROSCOPE_BIAS_PRIOR_STD     = 7200 * D2R / 3600; // 7200 deg/hr
    const double ACCELEROMETER_BIAS_PRIOR_STD = 20000 * 1.0e-5;    // 20000 mGal

    //Control 可观性参数
    double min_parallax_spd_ = 0.25;
    double max_good_parallax_count_ = 65;
    double noise_bound_sq_ = 0.00008;
    double reprojection_error_std_scale_ = 1.25;

    // HessianStability 参数
    double last_condition_number_ = 0.0;
    int stable_check_count_ = 0;

    // 优化参数, 使用deque容器管理, 移除头尾不会造成数据内存移动
    // The state data in the sliding window
    std::deque<std::shared_ptr<PreintegrationBase>> preintegrationlist_;
    std::deque<IntegrationStateData> statedatalist_;
    std::deque<GNSS> gnsslist_;
    std::deque<double> timelist_;
    std::unordered_map<ulong, double> invdepthlist_;

    //tmp change
    std::atomic<bool> first_frame_pushed_{false}; // 表示第一个 frame 是否已成功入队
    std::unordered_map<uint32_t, double> exinvdepthlist_;
    std::vector<double*> pose_params_;
    std::vector<double*> mix_params_;

    double extrinsic_[8]{0};

    //tmp change
    std::deque<std::shared_ptr<PreintegrationBase>> sub_preintegrationlist_;
    std::deque<IntegrationStateData> sub_statedatalist_;
    std::deque<double> sub_timelist_;
    std::unordered_map<ulong, double> sub_invdepthlist_;

    std::vector<double> unused_time_nodes_;

    // 边缘化
    // Marginalization variables
    std::shared_ptr<MarginalizationInfo> last_marginalization_info_{nullptr};
    std::vector<double *> last_marginalization_parameter_blocks_;

    //tmp change
    std::vector<ceres::ResidualBlockId> imu_factor_blocks_;
    ceres::ResidualBlockId imu_error_block_;
    ceres::ResidualBlockId imu_pose_prior_block_;
    ceres::ResidualBlockId imu_mix_prior_block_;

    // 先验
    // The prior
    bool is_use_prior_{false};
    double mix_prior_[18];
    double mix_prior_std_[18];
    double pose_prior_[7];
    double pose_prior_std_[6];

    // 融合对象
    // GVINS fusion objects
    Tracking::Ptr tracking_;
    Map::Ptr map_;
    Camera::Ptr camera_;
    Drawer::Ptr drawer_;

    // 多线程
    // Multi-thread variables
    std::thread drawer_thread_;
    std::thread tracking_thread_;
    std::thread optimization_thread_;
    std::thread fusion_thread_;

    std::atomic<bool> isoptimized_{false};
    std::atomic<bool> isfinished_{false};
    std::atomic<bool> isgnssready_{false};
    std::atomic<bool> isframeready_{false};
    std::atomic<bool> isgnssobs_{false};
    std::atomic<bool> isvisualobs_{false};

    // IMU处理
    // Ins process
    std::mutex imu_buffer_mutex_;
    std::mutex fusion_mutex_;
    std::condition_variable fusion_sem_;
    std::mutex ins_mutex_;

    // 跟踪处理
    // Tracking process
    std::mutex frame_buffer_mutex_;
    std::mutex tracking_mutex_;
    std::condition_variable tracking_sem_;
    std::mutex keyframes_mutex_;

    //tmp change
    std::condition_variable frame_ready_cv_;
    std::condition_variable ins_cv_;
    std::mutex frame_ready_mutex_;

    // 优化处理
    // Optimization process
    std::mutex optimization_mutex_;
    std::mutex state_mutex_;
    std::condition_variable optimization_sem_;

    // 用于做 predicate（关键！）
    std::atomic<int> imu_pending_{0};
    std::atomic<int> frame_pending_{0};

    // 传感器数据
    // GVINS sensor data
    std::deque<Frame::Ptr> keyframes_;
    GNSS gnss_{0}, last_gnss_{0}, last_last_gnss_{0};
    Frame::Ptr first_frame_;

    std::queue<Frame::Ptr> frame_buffer_;
    double lastframetime{0};
    double curframetime{0};

    std::queue<IMU> imu_buffer_;
    std::deque<std::pair<IMU, IntegrationState>> ins_window_;

    // IMU参数
    // IMU parameters
    std::shared_ptr<IntegrationParameters> integration_parameters_;
    Preintegration::PreintegrationOptions preintegration_options_;
    IntegrationConfiguration integration_config_;

    double imudatarate_{200};
    double imudatadt_{0.005};
    size_t reserved_ins_num_;

    Vector3d antlever_;

    // 初始化信息
    // Initialization
    int initlength_;

    // 外参
    // Camera-IMU extrinsic
    Pose pose_b_c_;
    double td_b_c_;
    std::mutex extrinsic_mutex_;

    bool is_use_visualization_{true};

    // 优化选项
    // Optimization options
    bool optimize_estimate_extrinsic_;
    bool optimize_estimate_td_;
    double optimize_reprojection_error_std_;
    int optimize_num_iterations_;
    size_t optimize_windows_size_;

    double reprojection_error_std_;

    // 统计参数
    // Statistic variables
    int iterations_[2]{0};
    double timecosts_[3]{0};
    double outliers_[2]{0};

    // 文件IO
    // File IO
    FileSaver::Ptr navfilesaver_;
    FileSaver::Ptr imuerrfilesaver_;
    FileSaver::Ptr ptsfilesaver_;
    FileSaver::Ptr statfilesaver_;
    FileSaver::Ptr extfilesaver_;
    FileSaver::Ptr trajfilesaver_;

    // 系统状态
    // System state
    std::atomic<GVINSState> gvinsstate_{GVINS_ERROR};
};

#endif // GVINS_GVINS_H
