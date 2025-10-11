# 代码解读


1. 奖励函数
奖励函数的描述位于 `source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py` 的 `RewardsCfg` 类中。共9个。

    - undesired_contacts
        
        作用：用于检测四肢末端是否有正常的接触

    - motion_global_root_pos

        作用： 用于奖励机器人base在世界系下的位置跟随情况

    - motion_global_root_ori

        作用： 用于奖励机器人base在世界系下的姿态跟随情况

    - motion_body_pos

        作用： 用于奖励机器人所有的link在base下的位置跟随情况

    - motion_body_ori

        作用： 用于奖励机器人所有的link在base下的姿态跟随情况

    - motion_body_lin_vel

        作用： 用于奖励机器人base在世界系下的速度跟随情况

    - motion_body_ang_vel
        
        作用： 用于奖励机器人base在世界系下的角速度跟随情况

    - action_rate_l2

        作用： mean square计算action的速度变化率
    
    - joint_limit

        作用： 惩罚超出soft_joint_pos_limits范围的关节运动 

2. observation
