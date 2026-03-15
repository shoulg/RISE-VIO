# import pandas as pd
# import subprocess
# import numpy as np
# import os

# # ==== 只需要修改这一行 ====
# csv_file = "/sad/catkin_ws/src/rb_vins/tmp_output/251011/T20251021015653/trajectory.csv"
# # ==========================

# # 自动生成输出路径（同目录 + "_adjusted" 后缀）
# base, ext = os.path.splitext(csv_file)
# output_csv_file = f"{base}_adjusted{ext}"

# # Unix 时间转 GPS 时间
# def unix_to_gps(unix_time):
#     GPS_LEAP_SECOND = 18  # 根据数据情况调整
#     gps_time = unix_time + GPS_LEAP_SECOND - 315964800
#     gps_week = int(gps_time // 604800)
#     gps_sow = gps_time - gps_week * 604800
#     return gps_sow  # 只返回 GPS 周秒

# # 检查文件是否存在
# if not os.path.exists(csv_file):
#     print(f"❌ CSV file does not exist: {csv_file}")
#     exit(1)

# # 读取 CSV
# df = pd.read_csv(csv_file, sep=r"\s+", header=None)

# # # 若需要转换为 GPS 时间可取消注释：
# # df[0] = df[0].apply(unix_to_gps)

# # 选取列并调整顺序（TUM 格式）
# df_tum = df[[0, 1, 2, 3, 4, 5, 6, 7]]

# # 保存新文件
# df_tum.to_csv(output_csv_file, header=False, index=False, sep=" ")
# print(f"✅ Adjusted CSV saved to: {output_csv_file}")

# # 调用 evo_traj 绘图
# proc = subprocess.Popen(
#     ["evo_traj", "tum", output_csv_file, "--plot"],
#     stdin=subprocess.PIPE,
#     text=True
# )
# proc.wait()





# # 设置地面真值轨迹文件路径和估计轨迹文件路径
# ground_truth_file = "/sad/dataset/EuRoC/MH_05_difficult/mav0/state_groundtruth_estimate0/data.tum"
# estimated_trajectory_file = "/sad/catkin_ws/src/rb_vins/tmp_output/251011/T20251021015653/trajectory_adjusted.csv"
# # urban38

# # euroc
# # MH_01_easy

# # MH_03_medium

# # MH_05_difficult
# #/sad/catkin_ws/src/rb_vins/tmp_output/251011/T20251017025558/trajectory_adjusted.csv

# # V1_02_medium

# # V2_02_medium

# # V2_03_difficult



# # 检查文件是否存在
# if not os.path.exists(ground_truth_file):
#     print(f"Ground truth file does not exist: {ground_truth_file}")
#     exit(1)

# if not os.path.exists(estimated_trajectory_file):
#     print(f"Estimated trajectory file does not exist: {estimated_trajectory_file}")
#     exit(1)

# # 使用 subprocess 调用 evo 进行轨迹评估
# proc = subprocess.Popen(
#     [
#         "evo_ape",  # 轨迹评估命令
#         "tum",  # 指定轨迹格式为 TUM
#         ground_truth_file,  # 地面真值轨迹文件
#         estimated_trajectory_file,  # 估计轨迹文件
#         "--align",
#         "--correct_scale",
#         "-r", "trans_part",
#         "--plot",  # 绘制评估结果
#         # "--save_plot", "/sad/catkin_ws/src/rb_vins/myoutout/250901/V2_03_difficult.png",  # 保存成文件
#     ],
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     text=True
# )
# # proc = subprocess.Popen(
# #     [
# #         "evo_traj",  # 轨迹评估命令
# #         "tum",  # 指定轨迹格式为 TUM
# #         estimated_trajectory_file,  # 估计轨迹文件
# #         "--ref", ground_truth_file,  # 地面真值轨迹文件
# #         "-p",
# #         "--align",
# #         "--correct_scale",
# #         "--plot",  # 绘制评估结果
# #         # "--save_plot", "/sad/catkin_ws/src/rb_vins/myoutout/250901/V2_03_difficult.png",  # 保存成文件
# #     ],
# #     stdout=subprocess.PIPE,
# #     stderr=subprocess.PIPE,
# #     text=True
# # )

# # 获取命令的输出和错误信息
# stdout, stderr = proc.communicate()

# # 输出结果
# if proc.returncode == 0:
#     print("Trajectory evaluation completed successfully.")
#     print(stdout)
# else:
#     print("Error occurred during trajectory evaluation:")
#     print(stderr)










import os
import shutil
import subprocess
from datetime import datetime
import pandas as pd
import sys

# ========== 配置 ==========
csv_file = "/sad/catkin_ws/src/rb_vins/tmp_output/260202/T20260202151124/trajectory.csv"
data_index = "MH_04_difficult"
init_method = "new_200_gnc_drt_init"
out_root = "/sad/catkin_ws/src/rb_vins/myoutput/251010"
evo_cmd = "evo_ape"
AUTO_UNIX_TO_GPS = False

# 根据 data_index 自动生成地面真值路径
ground_truth_file = f"/sad/dataset/EuRoC/{data_index}/mav0/state_groundtruth_estimate0/data.tum"
# =================================

# 创建带时间戳的输出目录
ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir_base = os.path.join(out_root, data_index, init_method)
out_dir = os.path.join(out_dir_base, ts_str)
os.makedirs(out_dir, exist_ok=True)

# 文件路径
adjusted_csv = os.path.join(out_dir, f"trajectory_adjusted.csv")
save_plot_path = os.path.join(out_dir, f"{data_index}-{init_method}.png")
saved_vio_path = os.path.join(out_dir, f"{data_index}-{init_method}-vio.txt")
log_path = os.path.join(out_dir, f"{data_index}-{init_method}-evo.log")

def unix_to_gps(unix_time, leap=18):
    gps_time = unix_time + leap - 315964800.0
    gps_week = int(gps_time // 604800.0)
    return gps_time - gps_week * 604800.0

def die(msg, code=1):
    print(f"[ERROR] {msg}")
    sys.exit(code)

def main():
    if shutil.which(evo_cmd) is None:
        die(f"找不到 `{evo_cmd}`，请安装 evo 或把 evo_cmd 改为完整路径。")
    if not os.path.exists(csv_file):
        die(f"CSV 文件不存在：{csv_file}")
    if not os.path.exists(ground_truth_file):
        die(f"地面真值文件不存在：{ground_truth_file}")

    # 读取 CSV
    try:
        df = pd.read_csv(csv_file, sep=r"\s+", header=None, engine="python")
    except Exception as e:
        die(f"读取 CSV 失败: {e}")

    if df.shape[1] < 8:
        die(f"输入文件列数少于8（{df.shape[1]}），无法生成 TUM-like 文件。")

    # Unix -> GPS（可选）
    if AUTO_UNIX_TO_GPS:
        try:
            ts = df.iloc[:,0].astype(float)
            if ts.median() > 1e12:  # 毫秒时间戳
                ts /= 1000.0
            df.iloc[:,0] = ts.apply(unix_to_gps)
        except Exception:
            pass

    # 保存 adjusted CSV
    df.iloc[:, :8].astype(float).to_csv(adjusted_csv, header=False, index=False, sep=" ", float_format="%.9f")
    shutil.copy2(adjusted_csv, saved_vio_path)

    # evo_ape 命令
    cmd = [
        evo_cmd, "tum",
        ground_truth_file,
        saved_vio_path,
        "--correct_scale",
        "-r", "trans_part",
        "-va",
        "--save_plot", save_plot_path,
    ]

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    # 写日志
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# evo run: {datetime.now().isoformat()}\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write(proc.stdout or "")

    # 打印 evo_ape 完整输出
    print(proc.stdout or "<no output>")

    # 成功/失败提示
    if proc.returncode == 0:
        print(f"[OK] 轨迹评估完成。图片：{save_plot_path}  日志：{log_path}")
        return 0
    else:
        print(f"[ERROR] evo_ape 失败，查看日志：{log_path}")
        return proc.returncode

if __name__ == "__main__":
    sys.exit(main())