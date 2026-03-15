import os
import shutil
import subprocess
from datetime import datetime
import pandas as pd
import sys

# ========== 配置 ==========
csv_file = "/sad/catkin_ws/src/rb_vins/tmp_output/viode/251220/T20251220022221/trajectory.csv"
data_index = "parking_lot"
data_index_1 = "3_mid"
init_method = "new_200_drt_init"
out_root = "/sad/catkin_ws/src/rb_vins/myoutput/251010/VIODE"
evo_cmd = "evo_ape"
AUTO_UNIX_TO_GPS = False

# 根据 data_index 自动生成地面真值路径
ground_truth_file = f"/sad/dataset/VIODE/{data_index}/state_groundtruth_estimate0/0_none.tum"
# =================================

# 创建带时间戳的输出目录
ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir_base = os.path.join(out_root, data_index, data_index_1, init_method)
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