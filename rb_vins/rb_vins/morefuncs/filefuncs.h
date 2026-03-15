#pragma once

#include <iostream>
#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <filesystem>  // C++17
namespace fs = std::filesystem;

// 在程序开始时执行一次
inline void initializeLogFile(const std::string& folderPath, const std::string& fileName) {
    namespace fs = std::filesystem;

    std::cout << "[init] target = " << (fs::absolute(folderPath) / fileName) << std::endl;

    if (!fs::exists(folderPath)) {
        if (!fs::create_directories(folderPath)) {
            std::cerr << "Error: Could not create directory " << folderPath << std::endl;
            return;
        }
    }

    std::ofstream file(fs::path(folderPath) / fileName, std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " 
                  << (fs::absolute(folderPath) / fileName) << std::endl;
        return;
    }
    std::cout << "[init] cleared file: " << (fs::absolute(folderPath) / fileName) << std::endl;
}

inline void appendLog(int i, double sum_w, double cost_, int num_obs, double mu,
                      const std::string& folderPath, const std::string& fileName) {
    std::ofstream file(folderPath + "/" + fileName, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file to write: " << folderPath + "/" + fileName << std::endl;
        return;
    }

    file << "Iteration: " << i << ", "
         << "Sum of Weights: " << sum_w << ", "
         << "Cost: " << cost_ << ", "
         << "Num_obs: " << num_obs << ", "
         << "mu: " << mu << "\n";
    file.close();
}

inline void appendLog(int i, double sum_w, double cost_, int num_obs, double diff, double mu,
                      const std::string& folderPath, const std::string& fileName) {
    std::ofstream file(folderPath + "/" + fileName, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file to write: " << folderPath + "/" + fileName << std::endl;
        return;
    }

    file << "Iteration: " << i << ", "
         << "Sum of Weights: " << sum_w << ", "
         << "Cost: " << cost_ << ", "
         << "Num_obs: " << num_obs << ", "
         << "Cost Difference: " << diff << ", "
         << "mu: " << mu << "\n";
    file.close();
}

inline void appendLog(int i, double sum_w, double sum_w_tmp, double cost_, int num_obs, double num_obs_long, double diff, double mu,
                      const std::string& folderPath, const std::string& fileName) {
    std::ofstream file(folderPath + "/" + fileName, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file to write: " << folderPath + "/" + fileName << std::endl;
        return;
    }

    file << "Iteration: " << i << ", "
         << "Sum of Weights: " << sum_w << ", "
         << "Sum of Weights tmp: " << sum_w_tmp << ", "
         << "Cost: " << cost_ << ", "
         << "Num_obs: " << num_obs << ", "
         << "Num_obs_long: " << num_obs_long << ", "
         << "Cost Difference: " << diff << ", "
         << "mu: " << mu << "\n";
    file.close();
}

inline void appendLog(double timestamp, double rot_deg, double trans_error, const std::string& folderPath, const std::string& fileName) {
    std::ofstream file(folderPath + "/" + fileName, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file to write: " << folderPath + "/" + fileName << std::endl;
        return;
    }

    // 设置输出为非科学计数法，并保留6位小数（你可以调整）
    file << std::fixed << std::setprecision(9);

    file << "Timestamp: " << timestamp << ", "
         << "Rot_deg: " << rot_deg << ", "
         << "Trans_error: " << trans_error << "\n";
    file.close();
}

inline void insertLogLine(const std::string& message,
                          const std::string& folderPath,
                          const std::string& fileName) {
    std::ofstream file(folderPath + "/" + fileName, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file to insert log line: " << folderPath + "/" + fileName << std::endl;
        return;
    }

    file << message << "\n";
    file.close();
}

inline void clearLogsFolder(const std::string& folder_path) {
    try {
        if (!fs::exists(folder_path)) return;
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            fs::remove_all(entry.path());
        }
        std::cout << "Cleared folder: " << folder_path << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error clearing folder: " << e.what() << std::endl;
    }
}