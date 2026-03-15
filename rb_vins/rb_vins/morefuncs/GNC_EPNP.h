#ifndef GNC_EPNP_H
#define GNC_EPNP_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <functional>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <memory>
#include <cmath>

#include "morefuncs/filefuncs.h"
#include "common/types.h"

// 为 std::tuple<int, int, int> 添加 hash 特化
namespace std {
template <>
struct hash<std::tuple<int, int, int>> {
    std::size_t operator()(const std::tuple<int, int, int>& key) const {
        std::size_t h1 = std::hash<int>()(std::get<0>(key));
        std::size_t h2 = std::hash<int>()(std::get<1>(key));
        std::size_t h3 = std::hash<int>()(std::get<2>(key));
        // 组合哈希值（你也可以使用 boost::hash_combine）
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};
}  // namespace std

class EPNPEstimator {
public:
    using Ptr = std::shared_ptr<EPNPEstimator>;

    // 工厂构造函数（推荐）
    static Ptr create(double gnc_epnp_noise) {
        return std::make_shared<EPNPEstimator>(gnc_epnp_noise);
    }

    // 构造函数
    explicit EPNPEstimator(double gnc_epnp_noise)
        : gnc_epnp_noise_(gnc_epnp_noise) {}

    void reset(std::vector<Eigen::Vector3d> pts_3, vector<double> w, std::vector<cv::Point2f> un_pts_2) {
        pts_3_ = std::move(pts_3);
        W_long_ = std::move(w);
        W_.assign(pts_3_.size(), 1.0);
        W_tmp_.resize(pts_3_.size());  // 确保 W_tmp_ 有足够空间
        for (size_t i = 0; i < pts_3_.size(); ++i) {
            W_tmp_[i] = W_[i] * W_long_[i];
        }
        un_pts_2_ = std::move(un_pts_2);
        pnp_succ = false;
        need_remove = false;

        pts_num_ = static_cast<size_t>(pts_3_.size());
        // compute_cws();
        // compute_cws_long();
        compute_cws_voxel();
        compute_as();
        compute_M();
    }

    void compute_cws() {
        cws_[0][0] = cws_[0][1] = cws_[0][2] = 0;

        for (size_t i = 0; i < pts_num_; i++) {
            cws_[0][0] += pts_3_[i][0];
            cws_[0][1] += pts_3_[i][1];
            cws_[0][2] += pts_3_[i][2];
        }
        for (size_t j = 0; j < 3; j++)
            cws_[0][j] /= pts_num_;

        Eigen::MatrixXd PW0(pts_num_, 3);
        for (size_t i = 0; i < pts_num_; i++) {
            PW0.row(i) << pts_3_[i][0] - cws_[0][0], pts_3_[i][1] - cws_[0][1], pts_3_[i][2] - cws_[0][2];
        }
        Eigen::Matrix3d PW0tPW0 = PW0.transpose() * PW0;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(PW0tPW0);
        if (eigensolver.info() != Eigen::Success) {
            std::cerr << "Eigen decomposition failed!" << std::endl;
            return;
        }

        // 提取特征值和特征向量
        Eigen::Vector3d D = eigensolver.eigenvalues();
        Eigen::Matrix3d U = eigensolver.eigenvectors();

        for (size_t i = 1; i < 4; i++) {
            // 这里只需要遍历后面3个控制点
            double k = sqrt(D[i - 1] / pts_num_);
            cws_[i][0] = cws_[0][0] + k * U(0, i - 1);
            cws_[i][1] = cws_[0][1] + k * U(1, i - 1);
            cws_[i][2] = cws_[0][2] + k * U(2, i - 1);
        }
    }

    void compute_cws_voxel() {
        // Step 1: 筛选长期特征点
        std::vector<Eigen::Vector3d> long_pts;
        for (size_t i = 0; i < pts_3_.size(); ++i) {
            if (W_long_[i] > 0.5) {
                long_pts.push_back(pts_3_[i]);
            }
        }

        // Step 1: 体素滤波
        double voxel_size = estimate_voxel_size(long_pts);
        std::vector<Eigen::Vector3d> filtered_pts = voxel_filter(long_pts, voxel_size);
        size_t num_pts = filtered_pts.size();
        if (num_pts < 4) {
            std::cerr << "Too few points after voxel filtering!" << std::endl;
            return;
        }

        // Step 2: 计算质心
        cws_[0][0] = cws_[0][1] = cws_[0][2] = 0;
        for (const auto& pt : filtered_pts) {
            cws_[0][0] += pt[0];
            cws_[0][1] += pt[1];
            cws_[0][2] += pt[2];
        }
        for (int j = 0; j < 3; ++j)
            cws_[0][j] /= num_pts;

        // Step 3: 中心化矩阵
        Eigen::MatrixXd PW0(num_pts, 3);
        for (size_t i = 0; i < num_pts; ++i) {
            PW0.row(i) = filtered_pts[i] - Eigen::Vector3d(cws_[0][0], cws_[0][1], cws_[0][2]);
        }

        // Step 4: 协方差矩阵 & 特征分解
        Eigen::Matrix3d PW0tPW0 = PW0.transpose() * PW0;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(PW0tPW0);
        if (eigensolver.info() != Eigen::Success) {
            std::cerr << "Eigen decomposition failed!" << std::endl;
            return;
        }

        Eigen::Vector3d D = eigensolver.eigenvalues();    // λ1 <= λ2 <= λ3
        Eigen::Matrix3d U = eigensolver.eigenvectors();   // 每列是特征向量

        // Step 5: 构造剩余3个控制点
        for (int i = 1; i < 4; ++i) {
            double k = sqrt(D[i - 1] / num_pts);
            for (int j = 0; j < 3; ++j)
                cws_[i][j] = cws_[0][j] + k * U(j, i - 1);
        }
    }

    void compute_cws_long() {
        cws_[0][0] = cws_[0][1] = cws_[0][2] = 0.0;

        size_t valid_num = 0;

        // 1. 先计算质心（均值）
        for (size_t i = 0; i < pts_num_; ++i) {
            if (W_long_[i] > 0.5) {
                const auto& pt = pts_3_[i];
                cws_[0][0] += pt[0];
                cws_[0][1] += pt[1];
                cws_[0][2] += pt[2];
                ++valid_num;
            }
        }

        for (int j = 0; j < 3; ++j)
            cws_[0][j] /= valid_num;

        // 2. 构造中心化后的 PW0 矩阵（每行是一个 3D 点减去均值）
        Eigen::MatrixXd PW0(valid_num, 3);
        size_t row = 0;

        for (size_t i = 0; i < pts_3_.size(); ++i) {
            if (W_long_[i] > 0.5) {
                const auto& pt = pts_3_[i];
                PW0.row(row++) << pt[0] - cws_[0][0],
                                pt[1] - cws_[0][1],
                                pt[2] - cws_[0][2];
            }
        }

        // 3. 协方差矩阵并特征分解
        Eigen::Matrix3d PW0tPW0 = PW0.transpose() * PW0;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(PW0tPW0);
        if (eigensolver.info() != Eigen::Success) {
            std::cerr << "Eigen decomposition failed!" << std::endl;
            return;
        }

        Eigen::Vector3d D = eigensolver.eigenvalues();
        Eigen::Matrix3d U = eigensolver.eigenvectors();

        // 4. 构造剩余三个控制点
        for (int i = 1; i < 4; ++i) {
            double k = sqrt(D[i - 1] / valid_num);
            cws_[i][0] = cws_[0][0] + k * U(0, i - 1);
            cws_[i][1] = cws_[0][1] + k * U(1, i - 1);
            cws_[i][2] = cws_[0][2] + k * U(2, i - 1);
        }
    }

    void compute_as() {
        as_.resize(pts_3_.size());
        Eigen::Matrix3d cc;
        cc << cws_[1][0] - cws_[0][0], cws_[2][0] - cws_[0][0], cws_[3][0] - cws_[0][0], 
              cws_[1][1] - cws_[0][1], cws_[2][1] - cws_[0][1], cws_[3][1] - cws_[0][1], 
              cws_[1][2] - cws_[0][2], cws_[2][2] - cws_[0][2], cws_[3][2] - cws_[0][2];
        Eigen::Matrix3d cc_inv = cc.inverse();
        for (size_t i = 0; i < pts_num_; i++) {
            Eigen::Vector3d cp;
            cp << pts_3_[i][0] - cws_[0][0], pts_3_[i][1] - cws_[0][1], pts_3_[i][2] - cws_[0][2];
            Eigen::Vector3d a;
            a = cc_inv * cp;
            as_[i][0] = 1.0f - a[0] - a[1] - a[2];
            as_[i][1] = a[0];
            as_[i][2] = a[1];
            as_[i][3] = a[2];
        }
    }

    void compute_M() {
        M_.resize(2 * pts_num_, 12);
        for (size_t i = 0; i < pts_num_; ++i) {
            M_.row(2 * i) << -as_[i][0], 0, un_pts_2_[i].x * as_[i][0], -as_[i][1], 0, un_pts_2_[i].x * as_[i][1], 
                            -as_[i][2], 0, un_pts_2_[i].x * as_[i][2], -as_[i][3], 0, un_pts_2_[i].x * as_[i][3];
            M_.row(2 * i + 1) << 0, -as_[i][0], un_pts_2_[i].y * as_[i][0], 0, -as_[i][1], un_pts_2_[i].y * as_[i][1], 
                                0, -as_[i][2], un_pts_2_[i].y * as_[i][2], 0, -as_[i][3], un_pts_2_[i].y * as_[i][3];
        }
    }

    void estimate() {
        Eigen::MatrixXd v4(12, 4);
        compute_v4(W_tmp_, M_, v4);

        Eigen::Matrix<double, 6, 10> l_6x10;
        compute_l_6x10(v4, l_6x10);

        Eigen::Matrix<double, 6, 1> rho;
        compute_rho(rho);

        double Betas[4][4], rep_errors[4];
        double Rs[4][3][3], ts[4][3];

        find_betas_approx_1(l_6x10, rho, Betas[1]);
        gauss_newton(l_6x10, rho, Betas[1]);
        rep_errors[1] = compute_R_and_t(v4, Betas[1], Rs[1], ts[1]);

        find_betas_approx_2(l_6x10, rho, Betas[2]);
        gauss_newton(l_6x10, rho, Betas[2]);
        rep_errors[2] = compute_R_and_t(v4, Betas[2], Rs[2], ts[2]);

        find_betas_approx_3(l_6x10, rho, Betas[3]);
        gauss_newton(l_6x10, rho, Betas[3]);
        rep_errors[3] = compute_R_and_t(v4, Betas[3], Rs[3], ts[3]);

        int N = 1;
        if (rep_errors[2] < rep_errors[1]) N = 2;
        if (rep_errors[3] < rep_errors[N]) N = 3;

        R_best << Rs[N][0][0], Rs[N][0][1], Rs[N][0][2], 
                  Rs[N][1][0], Rs[N][1][1], Rs[N][1][2], 
                  Rs[N][2][0], Rs[N][2][1], Rs[N][2][2];
        
        t_best << ts[N][0], ts[N][1], ts[N][2];

        compute_residuals(v4, Betas[N]);
    }

    void compute_v4(const std::vector<double> & W, const Eigen::MatrixXd & M, Eigen::MatrixXd & v4) {
        Eigen::MatrixXd W2(2 * pts_num_, 2 * pts_num_);
        W2.setZero();
        for (size_t i = 0; i < pts_num_; ++i) {
            W2(2 * i, 2 * i) = W[i];
            W2(2 * i + 1, 2 * i + 1) = W[i];
        }

        Eigen::MatrixXd MtWM(12, 12);
        // MtWM = M.transpose() * W2 * M;
        MtWM.noalias() = M.transpose() * W2 * M;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es (MtWM);

        // Eigen::MatrixXd e_vectors = es.eigenvectors();
        // v4.block (0, 0, 12, 4) = e_vectors.block (0, 0, 12, 4);
        v4.noalias() = es.eigenvectors().leftCols(4);

    }

    void compute_l_6x10(const Eigen::MatrixXd & v4, Eigen::Matrix<double, 6, 10> & l_6x10) {
        Eigen::Matrix<double, 4, 6> dv[3];
        for (size_t i = 0; i < 4; i++) {
            size_t a = 0, b = 1;
            for (size_t j = 0; j < 6; j++) {
                dv[0](i, j) = v4(3 * a, i) - v4(3 * b, i);
                dv[1](i, j) = v4(3 * a + 1, i) - v4(3 * b + 1, i);
                dv[2](i, j) = v4(3 * a + 2, i) - v4(3 * b + 2, i);
                b++;
                if (b > 3) {
                    a++;
                    b = a + 1;
                }
            }
        }

        for (size_t i = 0; i < 6; i++) {
            Eigen::Matrix<double, 3, 1> dv0, dv1, dv2, dv3;
            dv0 << dv[0](0, i), dv[1](0, i), dv[2](0, i);
            dv1 << dv[0](1, i), dv[1](1, i), dv[2](1, i);
            dv2 << dv[0](2, i), dv[1](2, i), dv[2](2, i);
            dv3 << dv[0](3, i), dv[1](3, i), dv[2](3, i);
            l_6x10(i, 0) = dv0.dot(dv0);
            l_6x10(i, 1) = 2.0 * dv0.dot(dv1);
            l_6x10(i, 2) = dv1.dot(dv1);
            l_6x10(i, 3) = 2.0 * dv0.dot(dv2);
            l_6x10(i, 4) = 2.0 * dv1.dot(dv2);
            l_6x10(i, 5) = dv2.dot(dv2);
            l_6x10(i, 6) = 2.0 * dv0.dot(dv3);
            l_6x10(i, 7) = 2.0 * dv1.dot(dv3);
            l_6x10(i, 8) = 2.0 * dv2.dot(dv3);
            l_6x10(i, 9) = dv3.dot(dv3);
        }
    }

    void compute_rho(Eigen::Matrix<double, 6, 1> & rho) {
        rho[0] = dist2(cws_[0], cws_[1]);
        rho[1] = dist2(cws_[0], cws_[2]);
        rho[2] = dist2(cws_[0], cws_[3]);
        rho[3] = dist2(cws_[1], cws_[2]);
        rho[4] = dist2(cws_[1], cws_[3]);
        rho[5] = dist2(cws_[2], cws_[3]);
    }

    double dist2(const double * p1, const double * p2) {
        return
            (p1[0] - p2[0]) * (p1[0] - p2[0]) +
            (p1[1] - p2[1]) * (p1[1] - p2[1]) +
            (p1[2] - p2[2]) * (p1[2] - p2[2]);
    }

    void find_betas_approx_1(const Eigen::MatrixXd &L_6x10, const Eigen::MatrixXd &Rho, double * betas) {
        Eigen::Matrix<double, 6, 4> L_6x4;
        for (size_t i = 0; i < 6; i++) {
            L_6x4(i, 0) = L_6x10(i, 0);
            L_6x4(i, 1) = L_6x10(i, 1);
            L_6x4(i, 2) = L_6x10(i, 3);
            L_6x4(i, 3) = L_6x10(i, 6);
        }

        // 使用 SVD 分解求解线性方程 L_6x4 * B4 = Rho
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(L_6x4, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix<double, 4, 1> B4 = svd.solve(Rho);

        if (B4(0) < 0) {
            betas[0] = std::sqrt(-B4(0));
            betas[1] = -B4(1) / betas[0];
            betas[2] = -B4(2) / betas[0];
            betas[3] = -B4(3) / betas[0];
        } else {
            betas[0] = std::sqrt(B4(0));
            betas[1] = B4(1) / betas[0];
            betas[2] = B4(2) / betas[0];
            betas[3] = B4(3) / betas[0];
        }
    }

    void find_betas_approx_2(const Eigen::MatrixXd &L_6x10, const Eigen::VectorXd &Rho, double * betas) {
        Eigen::Matrix<double, 6, 3> L_6x3;
        
        // 提取 L_6x10 的前 3 列到 L_6x3
        for (size_t i = 0; i < 6; i++) {
            L_6x3(i, 0) = L_6x10(i, 0);
            L_6x3(i, 1) = L_6x10(i, 1);
            L_6x3(i, 2) = L_6x10(i, 2);
        }

        // 使用 SVD 分解求解 L_6x3 * B3 = Rho
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(L_6x3, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Vector3d B3 = svd.solve(Rho);

        // 根据 B3 的值计算 betas
        if (B3(0) < 0) {
            betas[0] = sqrt(-B3(0));
            betas[1] = (B3(2) < 0) ? sqrt(-B3(2)) : 0.0;
        } else {
            betas[0] = sqrt(B3(0));
            betas[1] = (B3(2) > 0) ? sqrt(B3(2)) : 0.0;
        }

        if (B3(1) < 0) {
            betas[0] = -betas[0];
        }

        betas[2] = 0.0;
        betas[3] = 0.0;
    }

    void find_betas_approx_3(const Eigen::MatrixXd &L_6x10, const Eigen::VectorXd &Rho, double * betas) {
        Eigen::Matrix<double, 6, 5> L_6x5;

        // 提取 L_6x10 的前 5 列到 L_6x5
        for (size_t i = 0; i < 6; i++) {
            L_6x5(i, 0) = L_6x10(i, 0);
            L_6x5(i, 1) = L_6x10(i, 1);
            L_6x5(i, 2) = L_6x10(i, 2);
            L_6x5(i, 3) = L_6x10(i, 3);
            L_6x5(i, 4) = L_6x10(i, 4);
        }

        // 使用 SVD 分解求解 L_6x5 * B5 = Rho
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(L_6x5, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd B5 = svd.solve(Rho);

        // 根据 B5 的值计算 betas
        if (B5(0) < 0) {
            betas[0] = sqrt(-B5(0));
            betas[1] = (B5(2) < 0) ? sqrt(-B5(2)) : 0.0;
        } else {
            betas[0] = sqrt(B5(0));
            betas[1] = (B5(2) > 0) ? sqrt(B5(2)) : 0.0;
        }

        if (B5(1) < 0) {
            betas[0] = -betas[0];
        }

        betas[2] = B5(3) / betas[0];
        betas[3] = 0.0;
    }

    void gauss_newton(const Eigen::MatrixXd &L_6x10, const Eigen::VectorXd &Rho, double betas[4]) {
        const size_t iterations_number = 5;

        Eigen::Matrix<double, 6, 4> A;
        Eigen::Matrix<double, 6, 1> b;
        // Eigen::Matrix<double, 4, 1> x = Eigen::Matrix<double, 4, 1>::Zero();

        for (size_t k = 0; k < iterations_number; k++) {
            compute_A_and_b_gauss_newton(L_6x10, Rho, betas, A, b);
            
            // 使用QR分解求解线性方程组 A * x = b
            Eigen::VectorXd dx = A.colPivHouseholderQr().solve(b);
            
            for (size_t i = 0; i < 4; i++) {
                betas[i] += dx[i];
            }
        }
    }

    void compute_A_and_b_gauss_newton(const Eigen::MatrixXd &L_6x10, const Eigen::VectorXd &Rho, const double betas[4], 
                                        Eigen::Matrix<double, 6, 4> &A, Eigen::Matrix<double, 6, 1> &b) {
        for (size_t i = 0; i < 6; i++) {
            const Eigen::VectorXd rowL = L_6x10.row(i);
            
            A(i, 0) = 2 * rowL[0] * betas[0] +     rowL[1] * betas[1] +     rowL[3] * betas[2] +     rowL[6] * betas[3];
            A(i, 1) =     rowL[1] * betas[0] + 2 * rowL[2] * betas[1] +     rowL[4] * betas[2] +     rowL[7] * betas[3];
            A(i, 2) =     rowL[3] * betas[0] +     rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +     rowL[8] * betas[3];
            A(i, 3) =     rowL[6] * betas[0] +     rowL[7] * betas[1] +     rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

            // 计算 b[i]
            b(i) = Rho[i] -
                (rowL[0] * betas[0] * betas[0] +
                    rowL[1] * betas[0] * betas[1] +
                    rowL[2] * betas[1] * betas[1] +
                    rowL[3] * betas[0] * betas[2] +
                    rowL[4] * betas[1] * betas[2] +
                    rowL[5] * betas[2] * betas[2] +
                    rowL[6] * betas[0] * betas[3] +
                    rowL[7] * betas[1] * betas[3] +
                    rowL[8] * betas[2] * betas[3] +
                    rowL[9] * betas[3] * betas[3]);
        }
    }

    double compute_R_and_t(const Eigen::MatrixXd &v4, const double * betas, double R[3][3], double t[3]) {
        double ccs[4][3];
        compute_ccs(v4, betas, ccs);

        std::vector<std::array<double, 3>> pcs;
        compute_pcs(ccs, pcs);

        solve_for_sign(ccs, pcs);

        estimate_R_and_t(pcs, R, t);

        return reprojection_error(R, t);
    }

    void compute_ccs(const Eigen::MatrixXd &v4, const double * betas, double ccs[4][3]) {
        for (size_t i = 0; i < 4; i++)
            ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++)
                for (size_t k = 0; k < 3; k++)
                    ccs[j][k] += betas[i] * v4(3 * j + k, i);
        }
    }

    void compute_pcs(const double ccs[4][3], std::vector<std::array<double, 3>>  &pcs) {
        pcs.resize(pts_num_);
        for (size_t i = 0; i < pts_num_; i++) {
            pcs[i][0] = as_[i][0] * ccs[0][0] + as_[i][1] * ccs[1][0] + as_[i][2] * ccs[2][0] + as_[i][3] * ccs[3][0];
            pcs[i][1] = as_[i][0] * ccs[0][1] + as_[i][1] * ccs[1][1] + as_[i][2] * ccs[2][1] + as_[i][3] * ccs[3][1];
            pcs[i][2] = as_[i][0] * ccs[0][2] + as_[i][1] * ccs[1][2] + as_[i][2] * ccs[2][2] + as_[i][3] * ccs[3][2];
        }
    }

    void solve_for_sign(double ccs[4][3], std::vector<std::array<double, 3>>  &pcs) {
        if (pcs[0][2] < 0.0) {
            for (size_t i = 0; i < 4; i++)
                for (size_t j = 0; j < 3; j++)
                    ccs[i][j] = -ccs[i][j];

            for (size_t i = 0; i < pts_num_; i++) {
                pcs[i][0] = -pcs[i][0];
                pcs[i][1] = -pcs[i][1];
                pcs[i][2] = -pcs[i][2];
            }
        }
    }

    void estimate_R_and_t(const std::vector<std::array<double, 3>>  &pcs, double Rs[3][3], double ts[3]) {

        Eigen::Vector3d pc0_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d pw0_ = Eigen::Vector3d::Zero();
        double sum_w = 0;
        for (size_t i = 0; i < pts_num_; i++) {
            pc0_ += W_tmp_[i] * Eigen::Vector3d(pcs[i][0], pcs[i][1], pcs[i][2]);
            pw0_ += W_tmp_[i] * Eigen::Vector3d(pts_3_[i][0], pts_3_[i][1], pts_3_[i][2]);
            sum_w += W_tmp_[i];
        }
        pc0_ /= sum_w;
        pw0_ /= sum_w;

        Eigen::Matrix3d abt = Eigen::Matrix3d::Zero(); // 3x3 矩阵初始化为0
        // 遍历点对进行矩阵更新
        for (size_t i = 0; i < pts_num_; i++) {
            // 获取指向 pcs 和 pws 的指针
            Eigen::Vector3d pc(pcs[i][0], pcs[i][1], pcs[i][2]); // 将当前点 pcs[i] 转换为 Eigen::Vector3d
            Eigen::Vector3d pw(pts_3_[i][0], pts_3_[i][1], pts_3_[i][2]); // 将当前点 pws[i] 转换为 Eigen::Vector3d

            // 更新 abt 矩阵
            abt += W_tmp_[i] * (pw - pw0_) * (pc - pc0_).transpose(); // 进行矩阵外积并累加
        }

        // 对 abt 矩阵进行 SVD 分解
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(abt, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        R = V * U.transpose();

        // 检查旋转矩阵的行列式是否为正
        if (R.determinant() < 0) {
            U.col(2) *= -1;  // 将 U 的第三列取反以确保正的行列式
        }

        // 计算 t (平移向量)
        t = pc0_ - R * pw0_;

        for (size_t i = 0; i < 3; i++) 
            for (size_t j = 0; j < 3; j++) 
                Rs[i][j] = R(i, j);
        
        ts[0] = t[0];
        ts[1] = t[1];
        ts[2] = t[2];
    }

    double reprojection_error(const double R[3][3], const double t[3]) {
        double sum2 = 0.0;
        double sum_w = 0;

        for (size_t i = 0; i < pts_num_; i++) {
            double Xc = R[0][0] * pts_3_[i][0] + R[0][1] * pts_3_[i][1] + R[0][2] * pts_3_[i][2] + t[0];
            double Yc = R[1][0] * pts_3_[i][0] + R[1][1] * pts_3_[i][1] + R[1][2] * pts_3_[i][2] + t[1];
            double inv_Zc = 1.0 / (R[2][0] * pts_3_[i][0] + R[2][1] * pts_3_[i][1] + R[2][2] * pts_3_[i][2] + t[2]);
            double ue = Xc * inv_Zc;
            double ve = Yc * inv_Zc;
            double u = un_pts_2_[i].x, v = un_pts_2_[i].y;

            sum2 += W_tmp_[i] * sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
            sum_w += W_tmp_[i];
        }

        return sum2 / sum_w;
    }

    void compute_residuals(const Eigen::MatrixXd &v4, const double * betas) {
        double ccs[4][3];
        compute_ccs(v4, betas, ccs);

        std::vector<std::array<double, 3>> pcs;
        compute_pcs(ccs, pcs);

        solve_for_sign(ccs, pcs);

        compute_align_error(ccs);
    }

    void compute_align_error(const double ccs[4][3]) {
        Eigen::VectorXd vccs(12);
        vccs << ccs[0][0], ccs[0][1], ccs[0][2], ccs[1][0], ccs[1][1], ccs[1][2], ccs[2][0], ccs[2][1], ccs[2][2], ccs[3][0], ccs[3][1], ccs[3][2];
        Eigen::VectorXd align_error(2 * pts_num_);
        align_error = M_ * vccs;

        residuals_sq_.resize(1, pts_num_);
        for (size_t i = 0; i < pts_num_; ++i) {
            residuals_sq_(i) = align_error[2 * i] * align_error[2 * i] + align_error[2 * i + 1] * align_error[2 * i + 1];
        }
    }

    void gnc_estimate() {
        size_t max_iterations = 50;
        double noise_bound_sq = gnc_epnp_noise_;//0.025
        double mu = 1;
        double cost_ = 0;
        double sum_W = 0;
        double prev_cost = 0;
        double gnc_factor = 1.4;
        double cost_threshold = 1e-12;
        // bool patch = true;
        // int count = 0;

        insertLogLine("=== Start With Non-Guess ===", "/sad/catkin_ws/ex_logs", "epnp_changes.txt");

        // Loop for performing GNC-TLS
        for (size_t i = 0; i < max_iterations; ++i) {
            estimate();

            if (i == 0) {
                // Initialize rule for mu
                double max_residual = residuals_sq_.maxCoeff();
                mu = 1 / (2 * max_residual / noise_bound_sq - 1);
                // if (std::isinf(mu)) mu = 1e-6;  // 如果是inf，给mu赋值1e-6
                // Degenerate case: mu = -1 because max_residual is very small
                // i.e., little to none noise
                if (mu <= 0) {
                    // std::cout << "GNC-TLS terminated because maximum residual at initialization is very small." << std::endl;
                    pnp_succ = true;
                    break;
                }

            }

            // if (std::isinf(mu)) mu = 1e-6;  // 如果是inf，给mu赋值1e-6
            // Fix R and solve for weights in closed form
            double th1 = (mu + 1) / mu * noise_bound_sq;
            double th2 = mu / (mu + 1) * noise_bound_sq;
            cost_ = 0;
            for (size_t j = 0; j < pts_num_; ++j) {
                // Also calculate cost in this loop
                // Note: the cost calculated is using the previously solved weights
                cost_ += W_tmp_[j] * residuals_sq_(j);

                if (residuals_sq_(j) >= th1) {
                    W_[j] = 0;
                    W_tmp_[j] = 0;
                } else if (residuals_sq_(j) <= th2) {
                    W_[j] = 1;
                    W_tmp_[j] = 1;
                } else {
                    W_[j] = sqrt(noise_bound_sq * mu * (mu + 1) / residuals_sq_(j)) - mu;
                    W_tmp_[j] = W_[j];
                    assert(W_[j] >= 0 && W_[j] <= 1);
                }
            }

            // Calculate cost
            double cost_diff = std::abs(cost_ - prev_cost);

            sum_W = std::accumulate(W_.begin(), W_.end(), 0.0);  // 计算权重 W 和

            appendLog(i, sum_W, cost_, pts_num_, cost_diff, mu, "/sad/catkin_ws/ex_logs", "epnp_changes.txt");

            if (cost_diff < cost_threshold) {
                // std::cout << "GNC-TLS solver terminated due to cost convergence." << std::endl;
                // std::cout << "Cost diff: " << cost_diff << std::endl;
                // std::cout << "Iterations: " << i << std::endl;
                // break;

                pnp_succ = (sum_W >= 10 && cost_ < 1.0);
                need_remove = (sum_W != pts_num_);
                break;
            }

            // Increase mu
            mu = mu * gnc_factor;
            prev_cost = cost_;
        }

        // pnp_succ = pnp_succ || (sum_W >= 10 && cost_ < 1.0);
    }

    //如果有初始猜测
    void gnc_estimate(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
        size_t max_iterations = 50;
        double noise_bound_sq = gnc_epnp_noise_;//0.15//0.005
        double mu = 1;
        double cost_ = 0;
        double sum_W = 0;
        double prev_cost = 0;
        double gnc_factor = 1.3;
        double cost_threshold = 1e-12;

        insertLogLine("=== Start With IMU-Guess ===", "/sad/catkin_ws/ex_logs", "epnp_changes.txt");

        double ccsp[4][3];
        for (size_t i = 0; i < 4; i++) {
            ccsp[i][0] = R(0, 0) * cws_[i][0] + R(0, 1) * cws_[i][1] + R(0, 2) * cws_[i][2] + t[0];
            ccsp[i][1] = R(1, 0) * cws_[i][0] + R(1, 1) * cws_[i][1] + R(1, 2) * cws_[i][2] + t[1];
            ccsp[i][2] = R(2, 0) * cws_[i][0] + R(2, 1) * cws_[i][1] + R(2, 2) * cws_[i][2] + t[2];
        }
        compute_align_error(ccsp);
        double max_residual = residuals_sq_.maxCoeff();
        mu = 1 / (2 * max_residual / noise_bound_sq - 1);
        // if (std::isinf(mu)) mu = 1e-6;  // 如果是inf，给mu赋值1e-6
        // Degenerate case: mu = -1 because max_residual is very small
        // i.e., little to none noise
        if (mu <= 0) {
            // std::cout << "GNC-TLS terminated because maximum residual at initialization is very small." << std::endl;
            pnp_succ = true;
            return;
        }

        {
            double th1 = (mu + 1) / mu * noise_bound_sq;
            double th2 = mu / (mu + 1) * noise_bound_sq;
            for (size_t j = 0; j < pts_num_; ++j) {
                if (residuals_sq_(j) >= th1) {
                    W_[j] = 0;
                    W_tmp_[j] = 0;
                } else if (residuals_sq_(j) <= th2) {
                    W_[j] = 1;
                    W_tmp_[j] = W_long_[j];
                } else {
                    W_[j] = sqrt(noise_bound_sq * mu * (mu + 1) / residuals_sq_(j)) - mu;
                    W_tmp_[j] = W_[j] * W_long_[j];
                    assert(W_[j] >= 0 && W_[j] <= 1);
                }
            }

            mu = mu * gnc_factor;
        }

        for (size_t i = 1; i < max_iterations; ++i) {
            estimate();

            // if (std::isinf(mu)) mu = 1e-6;  // 如果是inf，给mu赋值1e-6
            // Fix R and solve for weights in closed form
            double th1 = (mu + 1) / mu * noise_bound_sq;
            double th2 = mu / (mu + 1) * noise_bound_sq;
            cost_ = 0;
            for (size_t j = 0; j < pts_num_; ++j) {
                // Also calculate cost in this loop
                // Note: the cost calculated is using the previously solved weights
                cost_ += W_tmp_[j] * residuals_sq_(j);

                if (residuals_sq_(j) >= th1) {
                    W_[j] = 0;
                    W_tmp_[j] = 0;
                } else if (residuals_sq_(j) <= th2) {
                    W_[j] = 1;
                    W_tmp_[j] = 1;
                } else {
                    W_[j] = sqrt(noise_bound_sq * mu * (mu + 1) / residuals_sq_(j)) - mu;
                    W_tmp_[j] = W_[j];
                    assert(W_[j] >= 0 && W_[j] <= 1);
                }
            }

            // Calculate cost
            double cost_diff = std::abs(cost_ - prev_cost);

            sum_W = std::accumulate(W_.begin(), W_.end(), 0.0);  // 计算权重 W 和
            double sum_W_tmp = std::accumulate(W_tmp_.begin(), W_tmp_.end(), 0.0);
            double sum_W_long = std::accumulate(W_long_.begin(), W_long_.end(), 0.0);

            appendLog(i, sum_W, sum_W_tmp, cost_, pts_num_, sum_W_long, cost_diff, mu, "/sad/catkin_ws/ex_logs", "epnp_changes.txt");

            if (cost_diff < cost_threshold) {
                // std::cout << "GNC-TLS solver terminated due to cost convergence." << std::endl;
                // std::cout << "Cost diff: " << cost_diff << std::endl;
                // std::cout << "Iterations: " << i << std::endl;
                // break;

                if (sum_W < 10) {
                    R_best = R;
                    t_best = t;
                }
                pnp_succ = (sum_W >= 10 && cost_ < 1.0);
                // pnp_succ = (cost_ < 1.0);
                need_remove = (sum_W != pts_num_);
                // pnp_succ = sum_W >= 10;
                break;
            }

            // Increase mu
            mu = mu * gnc_factor;
            prev_cost = cost_;
        }

        // pnp_succ = pnp_succ || (sum_W >= 10 && cost_ < 1.0);
    }

    std::vector<cv::Point2f> takePts2() {
        return std::move(un_pts_2_);
    }

    vector<float> takeStatus() {
        vector<float> status;
        if (pnp_succ) {
            for (const auto &w: W_) {
                float bin = (w >= 0.5f) ? 1.0 : 0.0;
                status.push_back(bin);
            }
        } else {
            status.resize(W_.size());
            std::fill(status.begin(), status.end(), 1.0);
        }
        return status;
    }

    vector<uint8_t> takeBinaryStatus() {
        vector<uint8_t> status;
        if (pnp_succ) {
            for (const auto &w: W_) {
                uint8_t bin = (w >= 0.5f) ? 1 : 0;
                status.push_back(bin);
            }
        } else {
            status.resize(W_.size());
            std::fill(status.begin(), status.end(), 1);
        }
        return status;
    }

    Pose takePose() {
        Pose pji;
        pji.R = R_best;
        pji.t = t_best;
        return pji;
    }

    bool takeneedremove() {
        return need_remove;
    }

    std::vector<Eigen::Vector3d> takecws() {
        std::vector<Eigen::Vector3d> cws;
        for (size_t i = 0; i < 4; i++) {
            cws.emplace_back(cws_[i][0], cws_[i][1], cws_[i][2]);
        }
        return cws;
    }

    Eigen::Matrix<double, 6, 6> computeSqrtInfoFromVisualPrior(int inliers, double rot_error_deg, double trans_error_m,
        double sigma_rot_rad = 5 * M_PI / 180.0,  // 默认5°
        double sigma_trans = 0.25                    // 默认20cm
    ) {
        // 动态权重计算
        const int max_inliers = 150;
        const int min_inliers = 30;
        double inlier_score = std::clamp((inliers - min_inliers) / double(max_inliers - min_inliers), 0.0, 1.0);
        double rot_penalty = std::exp(-rot_error_deg / 10.0);   // 惩罚旋转 > 10°
        double trans_penalty = std::exp(-trans_error_m / 0.5);  // 惩罚平移 > 0.5m

        double weight = inlier_score * rot_penalty * trans_penalty;
        weight = std::clamp(weight, 0.05, 1.0);  // 防止为0，避免优化失效

        // 构造 sqrt_info_ 矩阵
        Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::Matrix<double, 6, 6>::Zero();
        sqrt_info.topLeftCorner<3,3>() = (weight / sigma_rot_rad) * Eigen::Matrix3d::Identity();  // 旋转
        sqrt_info.bottomRightCorner<3,3>() = (weight / sigma_trans) * Eigen::Matrix3d::Identity(); // 平移

        return sqrt_info;
    }

    // 自动估计 voxel_size
    double estimate_voxel_size(const std::vector<Eigen::Vector3d>& points, double ratio = 0.02) {
        if (points.empty()) return 0.05;
        Eigen::Vector3d min_pt = points[0];
        Eigen::Vector3d max_pt = points[0];
        for (const auto& pt : points) {
            min_pt = min_pt.cwiseMin(pt);
            max_pt = max_pt.cwiseMax(pt);
        }
        double range = (max_pt - min_pt).norm();
        return std::max(0.001, range * ratio);  // 防止过小
    }

    // 对点进行体素滤波
    std::vector<Eigen::Vector3d> voxel_filter(const std::vector<Eigen::Vector3d>& points, double voxel_size) {
        using VoxelKey = std::tuple<int, int, int>;
        std::unordered_map<VoxelKey, std::vector<Eigen::Vector3d>> voxel_map;

        for (const auto& pt : points) {
            int ix = static_cast<int>(std::floor(pt[0] / voxel_size));
            int iy = static_cast<int>(std::floor(pt[1] / voxel_size));
            int iz = static_cast<int>(std::floor(pt[2] / voxel_size));
            voxel_map[{ix, iy, iz}].push_back(pt);
        }

        std::vector<Eigen::Vector3d> filtered_points;
        filtered_points.reserve(voxel_map.size());

        for (const auto& kv : voxel_map) {
            const auto& pts = kv.second;
            Eigen::Vector3d mean_pt(0.0, 0.0, 0.0);
            for (const auto& p : pts) {
                mean_pt += p;
            }
            mean_pt /= pts.size();
            filtered_points.push_back(mean_pt);
        }

        return filtered_points;
    }

private:
    double gnc_epnp_noise_;
    std::vector<Eigen::Vector3d> pts_3_; // 3D点
    std::vector<cv::Point2f> un_pts_2_;
    std::vector<double> W_;
    std::vector<double> W_long_;
    std::vector<double> W_tmp_;
    Eigen::Matrix<double, 1, Eigen::Dynamic> residuals_sq_;
    size_t pts_num_;
    double cws_[4][3];    // 3D坐标
    std::vector<std::array<double, 4>> as_;
    Eigen::MatrixXd M_;
    Eigen::Matrix3d R_best;
    Eigen::Vector3d t_best;
    bool pnp_succ;
    bool need_remove;
};

#endif