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

#include "common/logging.h"
#include "tracking/map.h"
#include "tracking/frame.h"
#include "tracking/mappoint.h"

#include "morefactors/visualprior_factor.h"

// 在源文件中初始化静态变量
std::atomic<drtSFMFeature::id_type> drtSFMFeature::global_id_counter {0};

void Map::insertKeyFrame(const Frame::Ptr &frame) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    // New keyframe
    latest_keyframe_ = frame;
    if (keyframes_.find(frame->keyFrameId()) == keyframes_.end()) {
        keyframes_.insert(make_pair(frame->keyFrameId(), frame));
    } else {
        keyframes_[frame->keyFrameId()] = frame;
    }

    // New mappoints
    auto &unupdated_mappoints = frame->unupdatedMappoints();
    for (const auto &mappoint : unupdated_mappoints) {
        if (landmarks_.find(mappoint->id()) == landmarks_.end()) {
            landmarks_.insert(make_pair(mappoint->id(), mappoint));
        } else {
            landmarks_[mappoint->id()] = mappoint;
        }
    }

    if (keyframes_.size() > window_size_) {
        is_window_full_ = true;
    }

    // tmp change
    double ts = frame->stamp();
    auto kf_id = frame->keyFrameId();
    // 如果不希望重复 key 覆盖旧值，可改成 multimap
    auto tt_it = timeToKeyframeId_.find(ts);
    if (tt_it == timeToKeyframeId_.end()) {
        timeToKeyframeId_.emplace(ts, kf_id);
    } else {
        // 已存在相同时间戳，按需覆盖或跳过
        tt_it->second = kf_id;
    }
}

vector<ulong> Map::orderedKeyFrames() {
    std::unique_lock<std::mutex> lock(map_mutex_);

    vector<ulong> keyframeid;
    for (auto &keyframe : keyframes_) {
        keyframeid.push_back(keyframe.first);
    }
    std::sort(keyframeid.begin(), keyframeid.end());

    return keyframeid;
}

Frame::Ptr Map::oldestKeyFrame() {
    std::unique_lock<std::mutex> lock(map_mutex_);

    auto oldest = orderedKeyFrames()[0];
    return keyframes_.at(oldest);
}

const Frame::Ptr &Map::latestKeyFrame() {
    std::unique_lock<std::mutex> lock(map_mutex_);

    return latest_keyframe_;
}

void Map::removeMappoint(MapPoint::Ptr &mappoint) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    mappoint->setOutlier(true);
    mappoint->removeAllObservations();
    if (landmarks_.find(mappoint->id()) != landmarks_.end()) {
        landmarks_.erase(mappoint->id());
    }
    mappoint.reset();
}

void Map::removeKeyFrame(Frame::Ptr &frame, bool isremovemappoint) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    if (isremovemappoint) {
        // 移除与关键帧关联的所有路标点
        vector<ulong> mappointid;
        auto features = frame->features();
        for (auto &feature : features) {
            auto mappoint = feature.second->getMapPoint();
            if (mappoint) {
                // 参考帧非边缘化帧, 不移除
                auto ref_frame = mappoint->referenceFrame();
                if (ref_frame != frame) {
                    continue;
                }
                mappointid.push_back(mappoint->id());
            }
        }
        for (auto id : mappointid) {
            auto landmark = landmarks_.find(id);
            if (landmark != landmarks_.end()) {
                auto mappoint = landmark->second;
                auto nextkid = mappoint->nextReferenceKid(id);
                auto tmp_frame = keyframes_.find(nextkid);
                if (nextkid == id || tmp_frame == keyframes_.end() || !mappoint->updateReferenceFrame(tmp_frame->second, camera_)) {
                    mappoint->removeAllObservations();
                    // 强制设置为outlier
                    mappoint->setOutlier(true);
                    landmarks_.erase(id);
                }
            }
        }
        frame->clearFeatures();
    }

    // tmp change
    timeToKeyframeId_.erase(frame->stamp());

    // 移除关键帧
    keyframes_.erase(frame->keyFrameId());
    frame.reset();
}

double Map::mappointObservedRate(const MapPoint::Ptr &mappoint) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    size_t num_keyframes = keyframes_.size();
    size_t num_observed  = 0;

    auto features = mappoint->observations();
    for (auto &feature : features) {
        auto feat = feature.second.lock();
        if (!feat) {
            continue;
        }
        auto frame = feat->getFrame();
        if (!frame) {
            continue;
        }

        if (keyframes_.find(frame->keyFrameId()) != keyframes_.end()) {
            num_observed += 1;
        }
    }
    return static_cast<double>(num_observed) / static_cast<double>(num_keyframes);
}

//drt_vio_init add
void Map::updateSFMConstruct(double cur_time, std::vector<drtSFMFeature>& construct) {
    double oldest_time = 0;
    {
        std::unique_lock<std::mutex> lock(map_mutex_);
        oldest_time = timeToKeyframeId_.begin()->first;
    }
    {
        std::unique_lock<std::mutex> lock(sfm_mutex_);

        // 遍历输入的 feature
        for (auto& feature : construct) {
            auto it = SFMConstruct_.find(feature.id);
            if (it != SFMConstruct_.end()) {
                auto& SFMFeature_obs = it->second.obs;

                while (!feature.obs.empty()) {
                    auto first_it = feature.obs.begin();
                    auto node = feature.obs.extract(first_it);
                    if (node) {
                        SFMFeature_obs.insert(std::move(node));
                    }
                }

                // 保持最近的 10 个观测
                while (SFMFeature_obs.size() > 11) {
                    SFMFeature_obs.erase(SFMFeature_obs.begin());
                }
            } else {
                // 移动插入新的 feature
                drtSFMFeature tmp_feature;
                tmp_feature.id = feature.id;
                tmp_feature.start_frame_ = feature.start_frame_;
                tmp_feature.state = feature.state;
                tmp_feature.p3d = feature.p3d;
                tmp_feature.obs = std::move(feature.obs);  // 移动 obs
                tmp_feature.estimated_depth = feature.estimated_depth;
                SFMConstruct_.emplace(feature.id, std::move(tmp_feature));

                // 显式清理 feature.obs，避免后续非法访问
                feature.obs.clear();
            }
        }

        // 移除早于滑窗观测的元素
        for (auto it = SFMConstruct_.begin(); it != SFMConstruct_.end();) {
            auto& obs = it->second.obs;

            // 移除早于 oldest_time 的观测
            for (auto obs_it = obs.begin(); obs_it != obs.end();) {
                if (obs_it->first < oldest_time) {
                    obs_it = obs.erase(obs_it);
                } else {
                    ++obs_it;
                }
            }

            // 如果观测数量不足或没有最新观测，移除该 feature
            if (obs.empty() || (obs.size() < 4 && obs.rbegin()->first != cur_time)) {
                it = SFMConstruct_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void Map::removesfmfeaturebyid(std::vector<uint32_t> &removesfm_ids) {
    std::unique_lock<std::mutex> lock(sfm_mutex_);

    // 遍历输入的 feature
    for (auto& id : removesfm_ids) {
        auto it = SFMConstruct_.find(id);
        if (it != SFMConstruct_.end()) {
            SFMConstruct_.erase(it);
        }
    }
}

void Map::retriangulate(const std::deque<std::pair<double, Pose>>& poselist) {
    {
        std::unique_lock<std::mutex> mapLock(map_mutex_);
        for (const auto &[pstamp, pose]: poselist) {
            for (auto& [kfId, keyframe] : keyframes_) {
                if (keyframe->stamp() == pstamp) {
                    keyframe->setPose(pose);
                    break;
                }
            }
        }
    }
    {
        std::unique_lock<std::mutex> mapLock(map_mutex_);
        size_t remove_num = 0, change_num = 0;
        for (auto it = landmarks_.begin(); it != landmarks_.end();) {
            auto landmark = it->second;
            auto obs = landmark->observations();
            bool landmarkErased = false;
            for (size_t i = 0; i + 1 < poselist.size(); ++i) {
                double time0 = poselist.at(i).first;
                double time1 = poselist.at(i + 1).first;
                auto it0 = obs.find(time0);
                auto it1 = obs.find(time1);
                if (it0 != obs.end() && it1 != obs.end()) {
                    const Pose pose0 = keyframes_.at(timeToKeyframeId_.at(time0))->pose();
                    const Pose pose1 = keyframes_.at(timeToKeyframeId_.at(time1))->pose();
                    Eigen::Matrix<double, 3, 4> T_c_w_0 = pose2Tcw(pose0).topRows<3>();
                    Eigen::Matrix<double, 3, 4> T_c_w_1 = pose2Tcw(pose1).topRows<3>();
                    auto feature0 = it0->second.lock();
                    auto feature1 = it1->second.lock();
                    if (!feature0 || !feature1) continue;
                    const auto& pp0 = feature0->keyPoint();
                    const auto& pp1 = feature1->keyPoint();
                    Vector3d pc0 = feature0->point();
                    Vector3d pc1 = feature1->point();
                    Vector3d pw;
                    triangulatePoint(T_c_w_0, T_c_w_1, pc0, pc1, pw);
                    //tmp change
                    if (!isGoodToTrack(pp0, pose0, pw, 1.5, 3.0) || !isGoodToTrack(pp1, pose1, pw, 1.5, 3.0)) {
                        landmark->removeAllObservations();
                        landmark->setOutlier(true);
                        it = landmarks_.erase(it);
                        landmarkErased = true;
                        ++remove_num;
                        break;
                    }
                    ++change_num;
                    break;
                }
            }
            if (!landmarkErased) {
                ++it;
            }
        }
        LOGI << "remove: " << remove_num << " change pw: " << change_num << " mappoint";
    }
    {
        std::unique_lock<std::mutex> mapLock(map_mutex_);
        std::unique_lock<std::mutex> sfmlock(sfm_mutex_);
        size_t add_num = 0;
        size_t reject_num = 0;
        size_t num1 = 0;
        for (auto it = SFMConstruct_.begin(); it != SFMConstruct_.end();) {
            auto &featureData = it->second;
            // if (featureData.obs.size() < 4) continue;
            num1++;
            // 找到第一个关键帧观测
            auto obsIt = std::find_if(featureData.obs.begin(), featureData.obs.end(),
                [&](const auto &p){ return timeToKeyframeId_.count(p.first); });
            if (obsIt == featureData.obs.end()) {
                ++it;
                continue;
            }

            double time0 = 0, time1 = 0;
            Vector3d pc0, pc1, pw;
            Vector2d velocity0, velocity1;
            cv::Point2f pp0, pp1;
            cv::Point2f pts0, pts1;
            Pose pose0, pose1;

            time0 = obsIt->first;
            assert(timeToKeyframeId_.count(time0));
            ulong kf0 = timeToKeyframeId_.at(time0);
            pc0   = obsIt->second.point();
            pp0   = obsIt->second.uv_undis();
            pts0  = obsIt->second.uv();
            pose0 = keyframes_.at(kf0)->pose();
            velocity0 = obsIt->second.velocity();
            ++obsIt;
            // 找到有足够视差的第二次观测
            while (obsIt != featureData.obs.end()) {
                // 跳过那些不在 keyframe map 里的
                if (timeToKeyframeId_.find(obsIt->first) == timeToKeyframeId_.end()) {
                    ++obsIt;
                    continue;
                }
                // 取候选第二次观测
                time1 = obsIt->first;
                pp1 = obsIt->second.uv_undis();
                pose1 = keyframes_.at(timeToKeyframeId_.at(time1))->pose();
            
                // 计算视差
                double parallax = keyPointParallax(pp0, pp1, pose0, pose1);
                if (parallax < 15) {
                    // 视差太小，跳过
                    ++obsIt;
                    continue;
                }
                // 如果视差够大，就跳出循环，用这一帧
                break;
            }

            if (obsIt == featureData.obs.end()) {
                ++it;
                continue;
            }
            time1 = obsIt->first;
            pc1   = obsIt->second.point();
            pp1   = obsIt->second.uv_undis();
            pts1  = obsIt->second.uv();
            ulong kf1 = timeToKeyframeId_.at(time1);
            pose1 = keyframes_.at(kf1)->pose();
            velocity1 = obsIt->second.velocity();
            ++obsIt;
            // 三角化并检测质量
            Eigen::Matrix<double, 3, 4> T_c_w_0 = pose2Tcw(pose0).topRows<3>();
            Eigen::Matrix<double, 3, 4> T_c_w_1 = pose2Tcw(pose1).topRows<3>();
            triangulatePoint(T_c_w_0, T_c_w_1, pc0, pc1, pw);
            //tmp change
            if (!isGoodToTrack(pp0, pose0, pw, 0.7, 3.0) || !isGoodToTrack(pp1, pose1, pw, 0.7, 3.0)) {//2.0 3.0
                reject_num++;
                ++it;
                continue;
            }
            // 创建地图点并插入
            auto pc       = camera_->world2cam(pw, pose0);
            double depth  = pc.z();
            auto mappoint = MapPoint::createMapPoint(keyframes_.at(kf0),
                                                        pw, pp0, depth, MAPPOINT_TRIANGULATED);

            auto feature = Feature::createFeature(keyframes_.at(kf1),
                                                    velocity1, pc1, pp1, pts1,
                                                    FEATURE_TRIANGULATED, time1);
            mappoint->addObservation(time1, feature);
            feature->addMapPoint(mappoint);
            keyframes_.at(kf1)->addFeature(mappoint->id(), feature);
            mappoint->increaseUsedTimes();

            feature = Feature::createFeature(keyframes_.at(kf0),
                                                velocity0, pc0, pp0, pts0,
                                                FEATURE_TRIANGULATED, time0);
            mappoint->addObservation(time0, feature);
            feature->addMapPoint(mappoint);
            keyframes_.at(kf0)->addFeature(mappoint->id(), feature);
            mappoint->increaseUsedTimes();

            for (auto &p : featureData.obs) {
                double t = p.first;
                if (t == time0 || t == time1) continue;
                if (!timeToKeyframeId_.count(t)) continue;
                ulong kft = timeToKeyframeId_.at(t);
                if (!keyframes_.count(kft)) continue;  
                auto f = p.second;    
                feature = Feature::createFeature(keyframes_.at(kft),
                                                 f.velocity(), f.point(), f.uv_undis(), f.uv(),
                                                 FEATURE_TRIANGULATED, t);
                mappoint->addObservation(t, feature);
                feature->addMapPoint(mappoint);
                keyframes_.at(kft)->addFeature(mappoint->id(), feature);
                mappoint->increaseUsedTimes();
            }

            landmarks_[mappoint->id()] = mappoint;
            ++add_num;

            it = SFMConstruct_.erase(it);
        }
        LOGI << "fram " << num1 << " sfmfeature, reject: " << reject_num << " mappoint, add: " << add_num << " mappoint";
    }
    // {
    //     std::unique_lock<std::mutex> mapLock(map_mutex_);
    //     ceres::Problem problem;
    //     // 参数缓冲区
    //     std::unordered_map<ulong, std::array<double,4>> rot_param; // quaternion wxyz
    //     std::unordered_map<ulong, std::array<double,3>> trans_param; // tx,ty,tz
    //     std::unordered_map<ulong, std::array<double,3>> point_param; // X,Y,Z
    //     // 填充初值
    //     for (const auto &[stamp, kfid] : timeToKeyframeId_) {
    //         auto kf = keyframes_.at(kfid);
    //         Pose pose = kf->pose();
    //         // 假设 Pose 能给出四元数和位移（若不是，按你工程的方式提取）
    //         Eigen::Matrix3d R_wc = pose.R;
    //         Eigen::Matrix3d R_cw = R_wc.transpose();          // world -> camera
    //         Eigen::Quaterniond q_cw(R_cw);                    // q (w,x,y,z)
    //         q_cw.normalize();
    //         Eigen::Vector3d t_cw = - R_cw * pose.t;           // t_cw = -R_cw * C
    //         rot_param[kfid] = {q_cw.w(), q_cw.x(), q_cw.y(), q_cw.z()};
    //         trans_param[kfid] = {t_cw.x(), t_cw.y(), t_cw.z()};
    //         // add parameter blocks
    //         problem.AddParameterBlock(rot_param[kfid].data(), 4);
    //         problem.SetManifold(rot_param[kfid].data(), new ceres::QuaternionManifold());
    //         problem.AddParameterBlock(trans_param[kfid].data(), 3);
    //     }
    //     for (auto it = landmarks_.begin(); it != landmarks_.end(); ++it) {
    //         auto landmark = it->second;
    //         auto mpid = it->first;
    //         Eigen::Vector3d pw = landmark->pos(); // 3D world point
    //         point_param[mpid] = {pw.x(), pw.y(), pw.z()};
    //         problem.AddParameterBlock(point_param[mpid].data(), 3);
    //     }
    //     // 固定第一个 keyframe 的位姿以去掉平移+旋转自由度（如果 metric_scale == false，还需固定另一个平移来定尺度）
    //     if (!timeToKeyframeId_.empty()) {
    //         ulong first_kfid = timeToKeyframeId_.begin()->second;
    //         problem.SetParameterBlockConstant(rot_param[first_kfid].data());
    //         problem.SetParameterBlockConstant(trans_param[first_kfid].data());
    //         if (1) {
    //             // 若视觉仍是 up-to-scale，锁定最后一帧平移确定尺度
    //             ulong last_kfid = std::prev(timeToKeyframeId_.end())->second;
    //             problem.SetParameterBlockConstant(trans_param[last_kfid].data());
    //         }
    //     }
    //     // BA 前投影统计（快速检查初值是否合理）
    //     {
    //         std::vector<double> errs;
    //         errs.reserve(1000);
    //         for (const auto &[mpid, landmark] : landmarks_) {
    //             Eigen::Vector3d pw = landmark->pos();
    //             const auto &obs_map = landmark->observations();
    //             for (const auto &pp : obs_map) {
    //                 double tstamp = pp.first;
    //                 if (!timeToKeyframeId_.count(tstamp)) continue;
    //                 ulong kfid = timeToKeyframeId_.at(tstamp);
    //                 if (!keyframes_.count(kfid)) continue;
    //                 auto kf = keyframes_.at(kfid);
    //                 Pose pose = kf->pose();
    //                 Eigen::Vector3d pc = camera_->world2cam(pw, pose); // 你已有的 world2cam
    //                 if (pc.z() <= 0) { errs.push_back(1e6); continue; }
    //                 double u = pc.x() / pc.z();
    //                 double v = pc.y() / pc.z();
    //                 cv::Point2f uv(u, v);
    //                 auto fptr = pp.second.lock();
    //                 if (!fptr) continue;
    //                 cv::Point2f point2d = fptr->point_2d();
    //                 double e = std::hypot(uv.x - point2d.x, uv.y - point2d.y);
    //                 errs.push_back(e);
    //             }
    //         }
    //         if (!errs.empty()) {
    //             std::sort(errs.begin(), errs.end());
    //             double sum = 0; for (double v: errs) sum += v;
    //             double mean = sum / errs.size();
    //             double median = errs[errs.size()/2];
    //             double maxe = errs.back();
    //             LOGI << "BA pre-proj stats mean/median/max: " << mean << " / " << median << " / " << maxe;
    //         } else {
    //             LOGI << "BA pre-proj: no observations found";
    //         }
    //     }
    //     // 添加重投影残差
    //     ceres::LossFunction* huber = new ceres::HuberLoss(1.0); // delta 可调
    //     int obs_cnt = 0;
    //     for (const auto &[mpid, landmark]: landmarks_) {
    //         const auto& obs_map = landmark->observations(); // 假设 map<time, weak_ptr<Feature>>
    //         for (const auto &p : obs_map) {
    //             double tstamp = p.first;
    //             auto fptr = p.second.lock();
    //             if (!fptr) continue;
    //             // 通过时间找到 keyframe id（你已有 timeToKeyframeId_）
    //             if (!timeToKeyframeId_.count(tstamp)) continue;
    //             ulong kfid = timeToKeyframeId_.at(tstamp);
    //             // 只对我们要优化的 keyframes 加残差
    //             if (rot_param.find(kfid) == rot_param.end()) continue;
    //             // get pixel observation
    //             cv::Point2f uv = fptr->point_2d(); // 假设是像素坐标
    //             {
    //                 auto &rq_init = rot_param[kfid];
    //                 auto &rt_init = trans_param[kfid];
    //                 auto &pp_init = point_param[mpid];
    //                 Eigen::Quaterniond q_init(rq_init[0], rq_init[1], rq_init[2], rq_init[3]);
    //                 Eigen::Matrix3d R_cw_init = q_init.toRotationMatrix();
    //                 Eigen::Vector3d t_cw_init(rt_init[0], rt_init[1], rt_init[2]);
    //                 Eigen::Vector3d pw_init(pp_init[0], pp_init[1], pp_init[2]);
    //                 Eigen::Vector3d pc_init = R_cw_init * pw_init + t_cw_init;
    //                 if (pc_init.z() <= 1e-6 || pc_init.z() > 1e4) {
    //                     LOGI << "Invalid depth get!!!";
    //                     continue; // skip bad / behind / too far
    //                 }
    //             }
    //             ceres::CostFunction* cost_function = ReprojectionError3D::Create(uv.x, uv.y);
    //             problem.AddResidualBlock(cost_function, NULL, rot_param[kfid].data(), trans_param[kfid].data(), 
    //                                     point_param[mpid].data());	                              
    //             ++obs_cnt;
    //         }
    //     }
    //     // Solver options
    //     ceres::Solver::Options options;
    //     options.linear_solver_type = ceres::SPARSE_SCHUR;
    //     options.max_num_iterations = 50;
    //     options.num_threads = 4;
    //     options.minimizer_progress_to_stdout = false;
    //     // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    //     ceres::Solver::Summary summary;
    //     ceres::Solve(options, &problem, &summary);
    //     // LOG
    //     LOGI << "BA obs: " << obs_cnt << " cost_before/after: " << summary.initial_cost << " / " << summary.final_cost
    //         << " iters: " << summary.num_successful_steps;
    //     // 判断收敛并更新关键帧、地图点
    //     if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 1e-3) {
    //         // 更新 keyframe pose（注意顺序：你工程里似乎用 q.inverse() 的惯例）
    //         for (const auto &[stamp, kfid] : timeToKeyframeId_) {
    //             if (rot_param.find(kfid) == rot_param.end() || trans_param.find(kfid) == trans_param.end()) continue;
    //             auto kf = keyframes_.at(kfid);
    //             auto &rq = rot_param[kfid];
    //             auto &rt = trans_param[kfid];
    //             Eigen::Quaterniond q_cw_opt(rq[0], rq[1], rq[2], rq[3]); // optimized q (world->camera)
    //             Eigen::Matrix3d R_cw_opt = q_cw_opt.toRotationMatrix();
    //             Eigen::Matrix3d R_wc_opt = R_cw_opt.transpose();
    //             Eigen::Vector3d t_cw_opt(rt[0], rt[1], rt[2]);
    //             Eigen::Vector3d C_opt = - R_wc_opt * t_cw_opt; // camera center in world
    //             Pose newPose;
    //             newPose.R = R_wc_opt; // keep Pose semantics: R_wc (camera->world)
    //             newPose.t = C_opt;    // camera center in world
    //             kf->setPose(newPose);
    //         }
    //         // 更新 map points（世界坐标）
    //         for (const auto &[mpid, landmark] : landmarks_) {
    //             if (point_param.find(mpid) == point_param.end()) continue;
    //             auto &pp = point_param[mpid];
    //             Eigen::Vector3d new_pw(pp[0], pp[1], pp[2]);
    //             landmark->updatePoseAndDepth(new_pw, camera_);
    //         }
    //         // return true;
    //     } else {
    //         LOGW << "BA did not converge: " << summary.BriefReport();
    //         // return false;
    //     }
    // }
}

void Map::retriangulate_old() {
    {
        using KeyframeSnap = struct { ulong id; double stamp; Pose pose; };
        struct LandmarkSnap {
            ulong id;
            MapPoint::Ptr mp; // 保留 shared_ptr，保证锁外生存期
            std::vector<std::pair<double, std::weak_ptr<Feature>>> observations; // 把观测转为 vector，避免 allocator 问题
            Vector3d pw;
        };

        // 快照容器
        std::vector<KeyframeSnap> keyframe_snaps;
        std::vector<LandmarkSnap> candidate_landmarks;
        keyframe_snaps.reserve(keyframes_.size());
        candidate_landmarks.reserve(landmarks_.size() / 8 + 8);

        // 1) 锁内快照 (短锁)
        {
            std::lock_guard<std::mutex> lk(map_mutex_);

            for (const auto &kv : keyframes_) {
                keyframe_snaps.push_back({kv.first, kv.second->stamp(), kv.second->pose()});
            }
            if (keyframe_snaps.size() < 2) return;
            std::sort(keyframe_snaps.begin(), keyframe_snaps.end(), [](const KeyframeSnap &a, const KeyframeSnap &b){
                return a.stamp < b.stamp;
            });

            // 快照 candidate landmarks（观测数 <= 3）
            for (const auto &lmkv : landmarks_) {
                const auto &lm_sp = lmkv.second;
                // 明确做拷贝（或局部拷贝），避免移动或直接引用内部容器（语义不明确）
                auto obs_copy = lm_sp->observations(); // 可能是 unordered_map<..., Eigen::aligned_allocator<...>>
                if (obs_copy.empty()) continue;
                if (obs_copy.size() > 3) continue;

                LandmarkSnap snap;
                snap.id = lmkv.first;
                snap.mp = lm_sp;                // 保存 shared_ptr 快照
                snap.pw = lm_sp->pos();

                // 将 obs_copy 的内容拷贝到 vector（避免 allocator 不匹配）
                snap.observations.reserve(obs_copy.size());
                for (const auto &p : obs_copy) {
                    snap.observations.emplace_back(p.first, p.second);
                }

                candidate_landmarks.push_back(std::move(snap));
            }
        } // 释放 map_mutex_

        // 2) 锁外逐个处理 candidate_landmarks（大量计算）
        std::vector<ulong> remove_ids;
        std::vector<std::pair<ulong, Vector3d>> update_id_pw;
        std::vector<std::pair<MapPoint::Ptr, Vector3d>> update_ptr_pw;
        std::vector<MapPoint::Ptr> remove_ptrs;

        size_t keep_num = 0, remove_num = 0, change_num = 0;

        // for (const auto &snap : candidate_landmarks) {
        //     bool decided = false;
        //     // 遍历相邻 keyframe pair
        //     for (size_t i = 0; i + 1 < keyframe_snaps.size(); ++i) {
        //         double t0 = keyframe_snaps[i].stamp;
        //         double t1 = keyframe_snaps[i+1].stamp;
        //         // 因为我们把 obs 放到 vector 中，这里用线性搜索精确匹配时间
        //         std::weak_ptr<Feature> wp0, wp1;
        //         for (const auto &op : snap.observations) {
        //             if (op.first == t0) { wp0 = op.second; break; }
        //         }
        //         for (const auto &op : snap.observations) {
        //             if (op.first == t1) { wp1 = op.second; break; }
        //         }
        //         if (wp0.expired() || wp1.expired()) continue;
        //         auto f0 = wp0.lock();
        //         auto f1 = wp1.lock();
        //         if (!f0 || !f1) continue;
        //         const auto &pp0 = f0->keyPoint();
        //         const auto &pp1 = f1->keyPoint();
        //         Vector3d pc0 = f0->point();
        //         Vector3d pc1 = f1->point();
        //         Vector3d pw = snap.pw;
        //         Pose pose0 = keyframe_snaps[i].pose;
        //         Pose pose1 = keyframe_snaps[i+1].pose;
        //         // 如果当前 pw 已经满足质量则保留
        //         if (isGoodToTrack(pp0, pose0, pw, 1.2, 3.0) &&
        //             isGoodToTrack(pp1, pose1, pw, 1.2, 3.0)) {
        //             ++keep_num;
        //             decided = true;
        //             break;
        //         }
        //         // 三角化（pc0/pc1 应为归一化相机坐标）
        //         Eigen::Matrix<double, 3, 4> T_c_w_0 = pose2Tcw(pose0).topRows<3>();
        //         Eigen::Matrix<double, 3, 4> T_c_w_1 = pose2Tcw(pose1).topRows<3>();
        //         Vector3d new_pw;
        //         triangulatePoint(T_c_w_0, T_c_w_1, pc0, pc1, new_pw);
        //         // 三角化质量检查
        //         if (!isGoodToTrack(pp0, pose0, new_pw, 1.2, 3.0) ||
        //             !isGoodToTrack(pp1, pose1, new_pw, 1.2, 3.0)) {
        //             remove_ids.push_back(snap.id);
        //             remove_ptrs.push_back(snap.mp);
        //             ++remove_num;
        //             decided = true;
        //             break;
        //         }
        //         // 标记更新：保存 shared_ptr + new pw
        //         update_ptr_pw.emplace_back(snap.mp, new_pw);
        //         update_id_pw.emplace_back(snap.id, new_pw);
        //         ++change_num;
        //         decided = true;
        //         break;
        //     } // end keyframe pairs
        //     if (!decided) ++keep_num;
        // } // end candidate_landmarks

        // helper: 多视角 DLT 三角化
        auto triangulateMultiViewDLT = [&](const std::vector<Pose> &poses,
                                        const std::vector<Feature::Ptr> &feats,
                                        Vector3d &out_pw) -> bool {
            const size_t N = feats.size();
            if (N < 2) return false;

            // 构造 A (2N x 4)
            Eigen::MatrixXd A(2 * N, 4);
            for (size_t i = 0; i < N; ++i) {
                Eigen::Matrix<double, 3, 4> P = pose2Tcw(poses[i]).topRows<3>();
                Vector3d pc = feats[i]->point(); // 假定为相机归一化坐标 (x,y,z)
                // 如果 pc.z() 为 0，则跳过（退化）
                if (fabs(pc.z()) < 1e-12) return false;
                double u = pc.x() / pc.z();
                double v = pc.y() / pc.z();

                // u * P3 - P1
                A.row(2*i)     = u * P.row(2) - P.row(0);
                // v * P3 - P2
                A.row(2*i + 1) = v * P.row(2) - P.row(1);
            }

            // SVD 求解最小奇异值对应的解 X
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
            Eigen::MatrixXd V = svd.matrixV();
            if (V.cols() < 1) return false;
            Eigen::Vector4d X = V.col(V.cols() - 1);
            if (fabs(X(3)) < 1e-12) return false; // 齐次分母退化

            out_pw = X.head<3>() / X(3);
            return true;
        };

        for (const auto &snap : candidate_landmarks) {
            // 收集所有有效观测与对应的 pose
            std::vector<Feature::Ptr> valid_feats;
            std::vector<Pose> valid_poses;
            valid_feats.reserve(snap.observations.size());
            valid_poses.reserve(snap.observations.size());

            for (const auto &op : snap.observations) {
                double t = op.first;
                auto wp = op.second;
                if (wp.expired()) continue;
                auto f = wp.lock();
                if (!f) continue;

                // 在 keyframe_snaps 中查找时间戳匹配的 pose
                auto it = std::find_if(
                    keyframe_snaps.begin(), keyframe_snaps.end(),
                    [&](const KeyframeSnap &k){ return fabs(k.stamp - t) < 1e-9; }
                );
                if (it == keyframe_snaps.end()) continue;

                valid_feats.push_back(f);
                valid_poses.push_back(it->pose);
            }

            if (valid_feats.size() < 2) { ++keep_num; continue; }

            // 如果当前 pw 已经满足所有视角，则保留
            bool good_now = true;
            for (size_t i = 0; i < valid_feats.size(); ++i) {
                if (!isGoodToTrack(valid_feats[i]->keyPoint(), valid_poses[i], snap.pw, 0.5, 3.0)) {
                    good_now = false; break;
                }
            }
            if (good_now) { ++keep_num; continue; }

            // 使用所有观测做 DLT 三角化
            Vector3d new_pw;
            bool tri_ok = triangulateMultiViewDLT(valid_poses, valid_feats, new_pw);
            if (!tri_ok) {
                // 三角化失败 -> 删除
                remove_ids.push_back(snap.id);
                remove_ptrs.push_back(snap.mp);
                ++remove_num;
                continue;
            }

            // 三角化后检查质量（所有视角）
            bool good_new = true;
            for (size_t i = 0; i < valid_feats.size(); ++i) {
                if (!isGoodToTrack(valid_feats[i]->keyPoint(), valid_poses[i], new_pw, 1.2, 3.0)) {
                    good_new = false; break;
                }
            }

            if (!good_new) {
                remove_ids.push_back(snap.id);
                remove_ptrs.push_back(snap.mp);
                ++remove_num;
                continue;
            }

            // 更新 new pw
            update_ptr_pw.emplace_back(snap.mp, new_pw);
            update_id_pw.emplace_back(snap.id, new_pw);
            ++change_num;
        }

        // 3) 锁内只做 erase（短锁）
        {
            std::lock_guard<std::mutex> lk(map_mutex_);
            for (auto id : remove_ids) {
                auto it = landmarks_.find(id);
                if (it != landmarks_.end()) {
                    landmarks_.erase(it);
                }
            }
            // 对于 update，我们不在此直接修改 map 内数据（以减少持锁时间）。
        }

        // 锁外完成耗时操作
        for (auto &mp_sp : remove_ptrs) {
            if (!mp_sp) continue;
            mp_sp->removeAllObservations();
            mp_sp->setOutlier(true);
        }
        for (auto &p : update_ptr_pw) {
            auto &mp_sp = p.first;
            const Vector3d &new_pw = p.second;
            if (!mp_sp) continue;
            mp_sp->updatePoseAndDepth(new_pw, camera_);
        }

        LOGI << "keep: " << keep_num << " remove: " << remove_num
            << " change pw: " << change_num << " mappoint";

    }
    // {
    //     std::unique_lock<std::mutex> mapLock(map_mutex_);
    //     std::unique_lock<std::mutex> sfmlock(sfm_mutex_);
    //     size_t add_num = 0;
    //     size_t reject_num = 0;
    //     size_t num1 = 0;
    //     for (auto it = SFMConstruct_.begin(); it != SFMConstruct_.end();) {
    //         auto& [id, featureData] = *it;
    //         auto &observations = featureData.obs;
    //         if (observations.size() < 4 || featureData.estimated_depth < 0) {
    //             ++it;
    //             continue;
    //         }
    //         num1++;
    //         auto obs_it = observations.begin();
    //         while (obs_it != observations.end() && !getFrameByTimeNoLock(obs_it->first)) {
    //             ++obs_it;
    //         }
    //         if (obs_it == observations.end()) {
    //             ++it;
    //             continue;
    //         }
    //         auto ref_frame = getFrameByTimeNoLock(obs_it->first);
    //         Vector3d pw = camera_->cam2world(obs_it->second.point() * featureData.estimated_depth, ref_frame->pose());
    //         double time0 = 0, time1 = 0;
    //         Vector3d pc0, pc1;
    //         Vector2d velocity0, velocity1;
    //         cv::Point2f pp0, pp1;
    //         cv::Point2f pts0, pts1;
    //         Pose pose0, pose1;
    //         time0 = obs_it->first;
    //         pc0   = obs_it->second.point();
    //         pp0   = obs_it->second.uv_undis();
    //         pts0  = obs_it->second.uv();
    //         ulong kf0 = timeToKeyframeId_.at(time0);
    //         pose0 = ref_frame->pose();
    //         velocity0 = obs_it->second.velocity();
    //         ++obs_it;
    //         while (obs_it != featureData.obs.end()) {
    //             time1 = obs_it->first;
    //             // 跳过那些不在 keyframe map 里的
    //             if (timeToKeyframeId_.count(time1) == 0) {
    //                 ++obs_it;
    //                 continue;
    //             }
    //             // 取候选第二次观测
    //             auto obs_frame = getFrameByTimeNoLock(time1);
    //             pp1 = obs_it->second.uv_undis();
    //             pose1 = obs_frame->pose();
    //             // 计算视差
    //             double parallax = keyPointParallax(pp0, pp1, pose0, pose1);
    //             if (parallax < 8) {
    //                 // 视差太小，跳过
    //                 ++obs_it;
    //                 continue;
    //             }
    //             // 如果视差够大，就跳出循环，用这一帧
    //             break;
    //         }
    //         if (obs_it == featureData.obs.end()) {
    //             ++it;
    //             continue;
    //         }
    //         time1 = obs_it->first;
    //         assert(timeToKeyframeId_.count(time1) && "time1 not in timeToKeyframeId_");
    //         pc1   = obs_it->second.point();
    //         pp1   = obs_it->second.uv_undis();
    //         pts1  = obs_it->second.uv();
    //         ulong kf1 = timeToKeyframeId_.at(time1);
    //         auto obs_frame = getFrameByTimeNoLock(time1);
    //         pose1 = obs_frame->pose();
    //         velocity1 = obs_it->second.velocity();
    //         ++obs_it;
    //         Eigen::Matrix<double, 3, 4> T_c_w_0 = pose2Tcw(pose0).topRows<3>();
    //         Eigen::Matrix<double, 3, 4> T_c_w_1 = pose2Tcw(pose1).topRows<3>();
    //         //tmp change
    //         if (!isGoodToTrack(pp0, pose0, pw, 1.5, 3.0) || !isGoodToTrack(pp1, pose1, pw, 1.5, 3.0)) {
    //             triangulatePoint(T_c_w_0, T_c_w_1, pc0, pc1, pw);
    //             if (!isGoodToTrack(pp0, pose0, pw, 1.5, 3.0) || !isGoodToTrack(pp1, pose1, pw, 1.5, 3.0)) {
    //                 ++it;
    //                 reject_num++;
    //                 continue;
    //             }
    //         }
    //         auto pc       = camera_->world2cam(pw, pose0);
    //         double depth  = pc.z();
    //         auto mappoint = MapPoint::createMapPoint(ref_frame, pw, pp0, depth, MAPPOINT_TRIANGULATED);
    //         auto feature = Feature::createFeature(obs_frame, velocity1, pc1, pp1, pts1, FEATURE_TRIANGULATED, time1);
    //         mappoint->addObservation(time1, feature);
    //         feature->addMapPoint(mappoint);
    //         obs_frame->addFeature(mappoint->id(), feature);
    //         mappoint->increaseUsedTimes();
    //         feature = Feature::createFeature(ref_frame, velocity0, pc0, pp0, pts0, FEATURE_TRIANGULATED, time0);
    //         mappoint->addObservation(time0, feature);
    //         feature->addMapPoint(mappoint);
    //         ref_frame->addFeature(mappoint->id(), feature);
    //         mappoint->increaseUsedTimes();
    //         while(obs_it != observations.end()) {
    //             time1 = obs_it->first;
    //             auto tmp_frame = getFrameByTimeNoLock(time1);
    //             if (tmp_frame) {  
    //                 auto f = obs_it->second;        
    //                 feature = Feature::createFeature(tmp_frame, f.velocity(), f.point(), f.uv_undis(), f.uv(),
    //                                                  FEATURE_TRIANGULATED, time1);
    //                 mappoint->addObservation(time1, feature);
    //                 feature->addMapPoint(mappoint);
    //                 tmp_frame->addFeature(mappoint->id(), feature);
    //                 mappoint->increaseUsedTimes();
    //             }
    //             ++obs_it;
    //         }
    //         landmarks_[mappoint->id()] = mappoint;
    //         ++add_num;
    //         it = SFMConstruct_.erase(it);
    //     }
    //     LOGI << "fram " << num1 << " sfmfeature, reject: " << reject_num << " mappoint, add: " << add_num << " mappoint";
    // }
}

Eigen::Matrix4d Map::pose2Tcw(const Pose &pose) {
    Eigen::Matrix4d Tcw;
    Tcw.setZero();
    Tcw(3, 3) = 1;

    Tcw.block<3, 3>(0, 0) = pose.R.transpose();
    Tcw.block<3, 1>(0, 3) = -pose.R.transpose() * pose.t;
    return Tcw;
}

void Map::triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
                           const Eigen::Vector3d &pc0, const Eigen::Vector3d &pc1, Eigen::Vector3d &pw) {
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();

    design_matrix.row(0) = pc0[0] * pose0.row(2) - pose0.row(0);
    design_matrix.row(1) = pc0[1] * pose0.row(2) - pose0.row(1);
    design_matrix.row(2) = pc1[0] * pose1.row(2) - pose1.row(0);
    design_matrix.row(3) = pc1[1] * pose1.row(2) - pose1.row(1);

    Eigen::Vector4d point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    pw                    = point.head<3>() / point(3);
}

bool Map::isGoodToTrack(const cv::Point2f &pp, const Pose &pose, const Vector3d &pw, double scale, double depth_scale) {
    // 当前相机坐标系
    Vector3d pc = camera_->world2cam(pw, pose);

    // 深度检查
    if (!isGoodDepth(pc[2], depth_scale)) {
        return false;
    }

    // 重投影误差检查
    if (camera_->reprojectionError(pose, pw, pp).norm() > 1.5 * scale) {
        return false;
    }

    return true;
}

bool Map::isGoodDepth(double depth, double scale) {
    return ((depth > MapPoint::NEAREST_DEPTH) && (depth < MapPoint::FARTHEST_DEPTH * scale));
}

double Map::keyPointParallax(const cv::Point2f &pp0, const cv::Point2f &pp1, const Pose &pose0,
                                  const Pose &pose1) {
    Vector3d pc0 = camera_->pixel2cam(pp0);
    Vector3d pc1 = camera_->pixel2cam(pp1);

    // 补偿掉旋转
    Vector3d pc01 = pose1.R.transpose() * pose0.R * pc0;

    // 像素大小
    return (pc01.head<2>() - pc1.head<2>()).norm() * camera_->focalLength();
}

void Map::moveKeyFrameToNonKeyFrame(const Frame::Ptr &frame) {
  auto id = frame->keyFrameId();
  // 从 keyframes_ 中“摘”出来
  auto it = keyframes_.find(id);
  if (it == keyframes_.end()) return;

  // 移动语义：把 shared_ptr 从 keyframes_ 转到 nonkeyframes_
  nonkeyframes_.emplace(id, std::move(it->second));
  keyframes_.erase(it);

  // 不调用 frame.reset()，也不修改它的其他数据
}

void Map::moveNewestKeyFrameToNonKeyFrame(const Frame::Ptr &frame) {
    if (nonkeyframes_.find(frame->keyFrameId()) == nonkeyframes_.end()) {
        nonkeyframes_.insert(make_pair(frame->keyFrameId(), frame));
    } else {
        nonkeyframes_[frame->keyFrameId()] = frame;
    }
}

void Map::clearSubMap() {
    nonkeyframes_.clear();
    std::cout << "Sub map cleared (nonkeyframes_)." << std::endl;
}

Frame::Ptr Map::getframebytime(double time) {
    std::unique_lock<std::mutex> mapLock(map_mutex_);
    return getFrameByTimeNoLock(time);
}

Frame::Ptr Map::getFrameByTimeNoLock(double time) {
    Frame::Ptr frame = nullptr;
    auto it = timeToKeyframeId_.find(time);
    if (it != timeToKeyframeId_.end()) {
        frame = keyframes_.at(it->second);
    }
    return frame;
}