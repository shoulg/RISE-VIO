import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

import os
import cv2

def find_closest_image(frame_ref_stamp, img_dir):
    """
    给定秒级时间戳frame_ref_stamp，和图像目录img_dir，
    返回该目录下最接近该时间戳的图像文件的完整路径。
    """
    frame_ref_ns = int(frame_ref_stamp * 1e9)  # 转纳秒整数
    
    # 获取该目录所有png文件名对应的纳秒整数列表
    files = os.listdir(img_dir)
    png_files = [f for f in files if f.endswith('.png')]
    
    # 解析文件名为纳秒整数
    timestamps = []
    for f in png_files:
        try:
            ts = int(f[:-4])  # 去掉'.png'，转为整数纳秒
            timestamps.append(ts)
        except ValueError:
            # 文件名不是时间戳格式，忽略
            pass
    
    if len(timestamps) == 0:
        raise ValueError(f"No valid png timestamp files found in {img_dir}")
    
    # 找到与frame_ref_ns差值最小的时间戳
    timestamps_np = np.array(timestamps)
    idx = np.argmin(np.abs(timestamps_np - frame_ref_ns))
    closest_ts = timestamps_np[idx]
    
    closest_filename = f"{closest_ts}.png"
    full_path = os.path.join(img_dir, closest_filename)
    
    return full_path

def parse_matched_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    frame_ref_stamp = None
    cws = []
    imu_rot = []
    imu_trans = None
    epnp_rot = None
    epnp_trans = None
    pdata = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# frame_ref_stamp:"):
            frame_ref_stamp = float(line.split(":")[1].strip())
            i += 1
            continue
        if line.startswith("cws"):
            parts = line.split()
            coords = list(map(float, parts[2:]))
            cws.append(coords)
            i += 1
            continue
        if line.startswith("# imu_rotation_matrix"):
            imu_rot = [list(map(float, lines[i + j].split())) for j in range(1, 4)]
            imu_rot = np.array(imu_rot)
            i += 4
            continue
        if line.startswith("# imu_translation_vector"):
            imu_trans = np.array(list(map(float, lines[i + 1].split())))
            i += 2
            continue
        if line.startswith("# epnp_rotation_matrix"):
            epnp_rot = [list(map(float, lines[i + j].split())) for j in range(1, 4)]
            epnp_rot = np.array(epnp_rot)
            i += 4
            continue
        if line.startswith("# epnp_translation_vector"):
            epnp_trans = np.array(list(map(float, lines[i + 1].split())))
            i += 2
            continue
        parts = line.split()
        nums = list(map(float, parts))
        if len(nums) >= 11:
            pdata.append(nums[:11])
        i += 1

    pdata = np.array(pdata)
    return frame_ref_stamp, np.array(cws), imu_rot, imu_trans, epnp_rot, epnp_trans, pdata

# 🔷 遍历文件夹中所有txt文件并排序
log_dir = "/sad/catkin_ws/ex_logs/matched_points/"
img_dir = "/sad/dataset/EuRoC/V2_03_difficult/mav0/cam0/data/"
txt_files = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
# 提取时间戳作为排序依据
txt_files.sort(key=lambda f: float(f[:-4]))

for fname in txt_files:
    full_path = os.path.join(log_dir, fname)
    frame_ref_stamp, cws, imu_rot, imu_trans, epnp_rot, epnp_trans, pdata = parse_matched_file(full_path)

    pt2d_map = pdata[:, 0:2]
    pt2d_matched = pdata[:, 2:4]
    pt2d_matched_norm = pdata[:, 4:6]
    pt3d_map = pdata[:, 6:9]
    w = pdata[:, 9]
    sta = pdata[:, 10]

    # 🔷 输出信息
    print("=" * 50)
    print(f"File: {fname}")
    print(f"Timestamp: {frame_ref_stamp}")
    print("CWS:\n", np.array(cws))
    print("IMU Rotation:\n", imu_rot)
    print("IMU Translation:\n", imu_trans)
    print("EPNP Rotation:\n", epnp_rot)
    print("EPNP Translation:\n", epnp_trans)
    print("Parsed points shape:", pdata.shape)
    print("First 3 pt2d_map:", pt2d_map[:3])
    print("First 3 pt2d_matched:", pt2d_matched[:3])
    print("=" * 50)

    # 读取图像
    closest_img_path = find_closest_image(frame_ref_stamp, img_dir)
    img = cv2.imread(closest_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # #四张图单独显示

    # # --- 3D点图1，颜色用w ---
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(pt3d_map[:, 0], pt3d_map[:, 1], pt3d_map[:, 2], c=w, cmap='viridis', label='3D points (w)')
    # plt.colorbar(sc, ax=ax, label='weight w')
    # ax.scatter(cws[:, 0], cws[:, 1], cws[:, 2], c='r', s=100, marker='^', label='CWS')
    # ax.set_title(f'3D points and CWS (color by w), ts={frame_ref_stamp}')
    # ax.legend()
    # plt.show()

    # # --- 3D点图2，颜色用sta ---
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(pt3d_map[:, 0], pt3d_map[:, 1], pt3d_map[:, 2], c=sta, cmap='coolwarm', label='3D points (sta)')
    # plt.colorbar(sc, ax=ax, label='state sta')
    # ax.scatter(cws[:, 0], cws[:, 1], cws[:, 2], c='g', s=100, marker='^', label='CWS')
    # ax.set_title(f'3D points and CWS (color by sta), ts={frame_ref_stamp}')
    # ax.legend()
    # plt.show()

    # # --- 2D点图3，颜色用w ---
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.imshow(img_rgb)
    # sc = ax.scatter(pt2d_map[:, 0], pt2d_map[:, 1], c=w, cmap='viridis', label='pt2d_map', s=10)
    # ax.scatter(pt2d_matched[:, 0], pt2d_matched[:, 1], c=w, cmap='viridis', marker='x', label='pt2d_matched', s=10)
    # plt.colorbar(sc, ax=ax, label='weight w')
    # for i in range(len(pt2d_map)):
    #     ax.plot([pt2d_map[i, 0], pt2d_matched[i, 0]], [pt2d_map[i, 1], pt2d_matched[i, 1]], c=cm.viridis(w[i]))
    # ax.set_title(f'Points on image (color by w), ts={frame_ref_stamp}')
    # ax.set_aspect('equal')
    # ax.legend()
    # plt.show()

    # # --- 2D点图4，颜色用sta ---
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.imshow(img_rgb)
    # sc = ax.scatter(pt2d_map[:, 0], pt2d_map[:, 1], c=sta, cmap='coolwarm', label='pt2d_map', s=10)
    # ax.scatter(pt2d_matched[:, 0], pt2d_matched[:, 1], c=sta, cmap='coolwarm', marker='x', label='pt2d_matched', s=10)
    # plt.colorbar(sc, ax=ax, label='state sta')
    # for i in range(len(pt2d_map)):
    #     ax.plot([pt2d_map[i, 0], pt2d_matched[i, 0]], [pt2d_map[i, 1], pt2d_matched[i, 1]], c=cm.coolwarm(sta[i]))
    # ax.set_title(f'Points on image (color by sta), ts={frame_ref_stamp}')
    # ax.set_aspect('equal')
    # ax.legend()
    # plt.show()

    # 拼图：四个子图放在一个窗口
    fig = plt.figure(figsize=(16, 12))  # 稍微缩小窗口

    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.2, wspace=0.1)  
    # ↑ 上面一行表示：上排图稍高（1.2:1.0），上下空隙拉开（hspace=0.4）

    # -------- 图1：3D点，颜色=w --------
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    sc1 = ax1.scatter(pt3d_map[:, 0], pt3d_map[:, 1], pt3d_map[:, 2], c=w, cmap='viridis', label='3D points (w)')
    fig.colorbar(sc1, ax=ax1, shrink=0.6, label='weight w')
    ax1.scatter(cws[:, 0], cws[:, 1], cws[:, 2], c='r', s=100, marker='^', label='CWS')
    ax1.set_title('3D points (color by w)', fontsize=12)
    ax1.legend(fontsize=10)

    # -------- 图2：3D点，颜色=sta --------
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    sc2 = ax2.scatter(pt3d_map[:, 0], pt3d_map[:, 1], pt3d_map[:, 2], c=sta, cmap='coolwarm', label='3D points (sta)')
    fig.colorbar(sc2, ax=ax2, shrink=0.6, label='state sta')
    ax2.scatter(cws[:, 0], cws[:, 1], cws[:, 2], c='g', s=100, marker='^', label='CWS')
    ax2.set_title('3D points (color by sta)', fontsize=12)
    ax2.legend(fontsize=10)

    # -------- 图3：2D图像，颜色=w，稍小 --------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(img_rgb)
    sc3 = ax3.scatter(pt2d_map[:, 0], pt2d_map[:, 1], c=w, cmap='viridis', label='pt2d_map', s=8)
    ax3.scatter(pt2d_matched[:, 0], pt2d_matched[:, 1], c=w, cmap='viridis', marker='x', label='pt2d_matched', s=8)
    for i in range(len(pt2d_map)):
        ax3.plot([pt2d_map[i, 0], pt2d_matched[i, 0]], [pt2d_map[i, 1], pt2d_matched[i, 1]], c=cm.viridis(w[i]), linewidth=0.5)
    fig.colorbar(sc3, ax=ax3, shrink=0.7, label='weight w')
    ax3.set_title('2D matches (color by w)', fontsize=12)
    ax3.set_aspect('equal')
    ax3.legend(fontsize=8)

    # -------- 图4：2D图像，颜色=sta，稍小 --------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(img_rgb)
    sc4 = ax4.scatter(pt2d_map[:, 0], pt2d_map[:, 1], c=sta, cmap='coolwarm', label='pt2d_map', s=8)
    ax4.scatter(pt2d_matched[:, 0], pt2d_matched[:, 1], c=sta, cmap='coolwarm', marker='x', label='pt2d_matched', s=8)
    for i in range(len(pt2d_map)):
        ax4.plot([pt2d_map[i, 0], pt2d_matched[i, 0]], [pt2d_map[i, 1], pt2d_matched[i, 1]], c=cm.coolwarm(sta[i]), linewidth=0.5)
    fig.colorbar(sc4, ax=ax4, shrink=0.7, label='state sta')
    ax4.set_title('2D matches (color by sta)', fontsize=12)
    ax4.set_aspect('equal')
    ax4.legend(fontsize=8)

    # -------- 全局标题 --------
    fig.suptitle(f"Matched Points Overview @ timestamp = {frame_ref_stamp}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
