# -*- coding: utf-8 -*-
"""
@File    : show_tracking.py
@Author  : hailiang
@Contact : thl@whu.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt

log = np.loadtxt('/sad/catkin_ws/src/rb_vins/tmp_output/viode/251217/T20251219024951/tracking.txt')
# urban38
# /sad/catkin_ws/src/rb_vins/output/T20250327090648/tracking.txt
# /sad/catkin_ws/src/rb_vins/output/T20250408022850/tracking.txt
# /sad/catkin_ws/src/rb_vins/output/T20250408124947/tracking.txt

# euroc
# MH_01_easy
# /sad/catkin_ws/src/rb_vins/tmp_output/T20250415021443/tracking.txt
# /sad/catkin_ws/src/rb_vins/tmp_output/T20250415033309/tracking.txt
# /sad/catkin_ws/src/rb_vins/tmp_output/T20250415064719/tracking.txt
# /sad/catkin_ws/src/rb_vins/tmp_output/MH_03_medium/T20250419080010/tracking.txt

# /sad/catkin_ws/src/rb_vins/tmp_output/T20250417123556/tracking.txt

# MH_03_medium
# /sad/catkin_ws/src/rb_vins/tmp_output/MH_03_medium/T20250419061946/tracking.txt
# /sad/catkin_ws/src/rb_vins/tmp_output/MH_03_medium/T20250422122538/tracking.txt
# /sad/catkin_ws/src/rb_vins/tmp_output/MH_03_medium/T20250624062800/tracking.txt


# /sad/catkin_ws/src/rb_vins/tmp_output/MH_03_medium/T20250703065238/tracking.txt


plt.figure('interval')
plt.plot(log[:, 0], log[:, 1])
plt.grid()
plt.title('Average %0.2lf s' % np.average(log[:, 1]))
plt.tight_layout()

plt.figure('parallax')
plt.plot(log[:, 0], log[:, 2])
plt.grid()
plt.title('Average %0.2lf pixel' % np.average(log[:, 2]))
plt.tight_layout()

plt.figure('translation')
plt.plot(log[:, 0], log[:, 3])
plt.grid()
plt.title('Average %0.2lf m' % np.average(log[:, 3]))
plt.tight_layout()

plt.figure('rotation')
plt.plot(log[:, 0], log[:, 4])
plt.grid()
plt.title('Average %0.2lf deg' % np.average(log[:, 4]))
plt.tight_layout()

plt.figure('mappoint')
plt.plot(log[:, 0], log[:, 5])
plt.grid()
plt.title('Average %0.2lf' % np.average(log[:, 5]))
plt.tight_layout()

plt.figure('tiemcost')
plt.plot(log[:, 0], log[:, 6])
plt.grid()
plt.title('Average %0.2lf ms' % np.average(log[:, 6]))
plt.tight_layout()

plt.show()
