import numpy as np

# 加载 .npz 文件
data = np.load("/home/hpx/HPX_LOCO_2/whole_body_tracking/deploy_mujoco/motion.npz")

# 查看文件中包含的数组键
print(data.files)  # 输出文件中的数组名称列表

print(data['body_names'].tolist().index('pelvis'))
pelvis_idx = data['body_names'].tolist().index('pelvis')

print(data['dof_names'])
power = np.abs(data['dof_velocities']) * np.abs(data['dof_torque'])

print(power.shape)
# import matplotlib.pyplot as plt
# # 为每列绘制折线图
# plt.figure(figsize=(10, 6))  # 设置图形大小
# n, m = power.shape
# for col in range(6):
#     plt.plot(power[::10, col], label=f'Column {col+1}')  # 绘制每列，添加标签

# # 添加标题和标签
# plt.title('Data Plot for Each Column')
# plt.xlabel('Row Index')
# plt.ylabel('Value')
# plt.legend()  # 显示图例
# plt.grid(True)  # 添加网格
# plt.show()

import pandas as pd
header = data['dof_names'].tolist()
# 转换为 DataFrame
df = pd.DataFrame(power, columns=header)

# 写入 CSV 文件
df.to_csv('joint_power_run.csv', index=False)