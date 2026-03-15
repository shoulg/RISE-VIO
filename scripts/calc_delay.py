#!/usr/bin/env python3
import rosbag
import numpy as np

bag_path = "pose_timing.bag"
topic = "/pose"

delays = []
with rosbag.Bag(bag_path, 'r') as bag:
    for _, msg, t in bag.read_messages(topics=[topic]):
        # t: bag record time (ROS time, sim time when /clock is used)
        # msg.header.stamp: message timestamp
        d = (t - msg.header.stamp).to_sec()
        delays.append(d)

delays = np.array(delays, dtype=float)
print("N =", len(delays))
print("mean   =", float(np.mean(delays)))
print("median =", float(np.percentile(delays, 50)))
print("p95    =", float(np.percentile(delays, 95)))
print("min    =", float(np.min(delays)))
print("max    =", float(np.max(delays)))
print("std    =", float(np.std(delays)))
