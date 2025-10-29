import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

from typing import Union, List, Dict, Tuple
from scipy.spatial.transform import Rotation as R

import glfw
import os
import onnxruntime as ort
import copy
import math
import matplotlib.pyplot as plt
import queue
import threading
import pickle
import pinocchio as pin
from pinocchio.utils import zero
from pinocchio.robot_wrapper import RobotWrapper

"""
 conda install pinocchio -c conda-forge
"""
from math_func import *
from motion_loader import MotionLoader
from video_recorder import VideoRecorder

np.set_printoptions(precision=16, linewidth=100, threshold=np.inf, suppress=True)

current_path = os.getcwd()


class cfg:
    simulator_dt = 0.002
    policy_dt = 0.02

    policy_type = "onnx"  # torch or onnx
    policy_path = (
        current_path
        + "/deploy_mujoco/deploy_policy/Q1/"
        # + "2025-09-24_10-36-23_h1_2_0923_03_v0_11000step.onnx"
        + "2025-10-28_16-24-20_Q1_251021_03_walk_120Hz_4500"
        + ".onnx"
    )
    asset_path = "/deploy_mujoco/assets/Q1P01"
    # mjcf_path = current_path + asset_path + "/h1_2_wo_hand.xml"
    mjcf_path = current_path + asset_path + "/mjcf/Q1_01.xml"
    urdf_path = current_path + asset_path + "/urdf/Q1P01_urdf_rl.urdf"
    motion_file = (
        current_path
        + "/deploy_mujoco/artifacts/Q1/"
        + "Q1_251021_03_saw_120Hz:v0"
        # + "251021_04_boxing_120Hz:v0"
        # + "251014_single_action_forward_walk:v0"
        + "/motion.npz"
    )
    only_leg_flag = False  # True, False
    with_wrist_flag = True  # True, False

    ###############################
    # stiffness and damping param #
    ###############################
    leg_P_gains = [90, 60, 330, 330, 36, 36] * 2
    leg_D_gains = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5] * 2

    pelvis_P_gains = [42.0]
    pelvis_D_gains = [1.0]

    arm_P_gains = [42.0, 42.0, 23.0, 23.0, 8.0, 8.0, 8.0] * (2)
    arm_D_gains = [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8] * (2)
    
    head_P_gains = [3.0,1.5]
    head_D_gains = [0.6,0.3]
    ########################
    # joint maximum torque #
    ########################
    leg_tq_max = [90.0, 60.0, 330.0, 330.0, 36.0, 36.0] * (2)
    pelvis_tq_max = [42.0]
    arm_tq_max = [42.0, 42.0, 23.0, 23.0, 8.3, 8.3, 8.3] * (2)
    head_tq_max = [2.52,1.26]

    #####################
    # joint default pos #
    #####################
    leg_default_pos = [0.0, 0.0, -0.18, 0.38, -0.2, 0.0] * (2)
    pelvis_default_pos = [0.0]
    arm_default_pos = [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0] + [
        0.0,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    head_default_pos = [0.0,0.0]
    # leg_default_pos = [0.0] * (12)
    # torso_default_pos = [0.0]
    # arm_default_pos = [0.0] * (14)

    ################
    # action param #
    ################
    action_clip = 10.0
    action_scale = 0.25

    action_num = 29
    print("action_num: ", action_num)
    #############
    # obs param #
    #############
    frame_stack = 1
    num_single_obs = 154

    ####################
    # motion play mode #
    ####################
    """
     if motion_play is true, robots in mujoco will set 
     qpos and qvel through the retargeting dataset 
    """
    motion_play = True  # False, True
    """
    if motion_play is true and sim_motion_play is true,
    robots in mujoco will set qpos and qvel through the 
    dataset recorded in isaac sim
    """
    sim_motion_play = True  # False, True,

    ###########################################
    # Data conversion of isaac sim and mujoco #
    ###########################################
    isaac_sim_joint_name = [
        "L_shoulder_pitch_joint",
        "R_shoulder_pitch_joint",
        "head_yaw_joint",
        "pelvis_joint",
        "L_shoulder_roll_joint",
        "R_shoulder_roll_joint",
        "head_pitch_joint",
        "L_hip_roll_joint",
        "R_hip_roll_joint",
        "L_shoulder_yaw_joint",
        "R_shoulder_yaw_joint",
        "L_hip_yaw_joint",
        "R_hip_yaw_joint",
        "L_elbow_joint",
        "R_elbow_joint",
        "L_hip_pitch_joint",
        "R_hip_pitch_joint",
        "L_forearm_yaw_joint",
        "R_forearm_yaw_joint",
        "L_knee_joint",
        "R_knee_joint",
        "L_wrist_roll_joint",
        "R_wrist_roll_joint",
        "L_ankle_pitch_joint",
        "R_ankle_pitch_joint",
        "L_wrist_yaw_joint",
        "R_wrist_yaw_joint",
        "L_ankle_roll_joint",
        "R_ankle_roll_joint",
    ]

    mujoco_joint_name = [
        "L_shoulder_pitch_joint",
        "L_shoulder_roll_joint",
        "L_shoulder_yaw_joint",
        "L_elbow_joint",
        "L_forearm_yaw_joint",
        "L_wrist_roll_joint",
        "L_wrist_yaw_joint",
        "R_shoulder_pitch_joint",
        "R_shoulder_roll_joint",
        "R_shoulder_yaw_joint",
        "R_elbow_joint",
        "R_forearm_yaw_joint",
        "R_wrist_roll_joint",
        "R_wrist_yaw_joint",

        "pelvis_joint",

        "L_hip_roll_joint",
        "L_hip_yaw_joint",
        "L_hip_pitch_joint",
        "L_knee_joint",
        "L_ankle_pitch_joint",
        "L_ankle_roll_joint",
        "R_hip_roll_joint",
        "R_hip_yaw_joint",
        "R_hip_pitch_joint",
        "R_knee_joint",
        "R_ankle_pitch_joint",
        "R_ankle_roll_joint",

        "head_yaw_joint",
        "head_pitch_joint",
    ]  # [joint.name for joint in self.spec.joints][1:]

    isaac_sim_link_name = [
        "torso_link",
        "L_shoulder_pitch_link",
        "R_shoulder_pitch_link",
        "head_yaw_link",
        "pelvis_link",
        "L_shoulder_roll_link",
        "R_shoulder_roll_link",
        "head_pitch_link",
        "L_hip_roll_link",
        "R_hip_roll_link",
        "L_shoulder_yaw_link",
        "R_shoulder_yaw_link",
        "L_hip_yaw_link",
        "R_hip_yaw_link",
        "L_elbow_link",
        "R_elbow_link",
        "L_hip_pitch_link",
        "R_hip_pitch_link",
        "L_forearm_yaw_link",
        "R_forearm_yaw_link",
        "L_knee_link",
        "R_knee_link",
        "L_wrist_roll_link",
        "R_wrist_roll_link",
        "L_ankle_pitch_link",
        "R_ankle_pitch_link",
        "L_wrist_yaw_link",
        "R_wrist_yaw_link",
        "L_ankle_roll_link",
        "R_ankle_roll_link",
    ]

    motion_body_names = [
        "pelvis_link",
        "L_hip_yaw_link",
        "L_knee_link",
        "L_ankle_roll_link",
        "R_hip_yaw_link",
        "R_knee_link",
        "R_ankle_roll_link",
        "torso_link",
        "L_shoulder_roll_link",
        "L_elbow_link",
        "L_wrist_yaw_link",
        "R_shoulder_roll_link",
        "R_elbow_link",
        "R_wrist_yaw_link",
        "head_pitch_link",
    ]

    motion_reference_body = "torso_link"


class pin_mj:
    def __init__(self, _URDF_PATH=""):
        # ========== 1. 准备Pinocchio模型 ==========
        URDF_PATH = _URDF_PATH
        self.robot: RobotWrapper = RobotWrapper.BuildFromURDF(
            URDF_PATH, current_path + cfg.asset_path, pin.JointModelFreeFlyer()
        )

        self.base_pos_world = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.base_quat_world = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def mujoco_to_pinocchio(
        self,
        joint_angles,
        base_pos=np.array([0.0, 0.0, 0.0], dtype=np.double),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.double),
    ):
        """
        将从Mujoco获取的机器人状态(基座位置、姿态、关节角)赋值到Pinocchio中。
        base_pos: np.array([x, y, z]) 基座在世界坐标系的位置
        base_quat: np.array([x, y, z, w]) 基座在世界坐标系的四元数 (Pinocchio默认的四元数顺序同为 [x,y,z,w])
        joint_angles: np.array([...]) 机器人关节角，长度为model.nq - 7(若有浮动基), 或 model.nq(若固定基)
        model, data: Pinocchio的model和data
        """

        q: np.ndarray = zero(
            self.robot.model.nq
        )  # 广义坐标 [7 + nJoints] (若 free-flyer)

        # 如果是浮动基模式，则前7维为 [x, y, z, q_x, q_y, q_z, q_w]
        # 注意：Pinocchio中free-flyer的顺序约定是 [xyz, qwxyz]
        # 若是固定基，则model.nq == 机器人关节数，无需设置基座
        if self.robot.model.joints[1].shortname() == "JointModelFreeFlyer":
            q[0:3] = base_pos
            q[3:7] = base_quat  # [x, y, z, w]
            # 后面是机器人关节
            q[7:] = joint_angles
        else:
            # 如果是固定基模型，则整段q都是关节
            q[:] = joint_angles

        # pin.forwardKinematics(self.model, self.data, q.astype(np.double).reshape(self.model.nq))
        self.robot.framesForwardKinematics(q)
        """ros的python环境在bash中会被引入，产生冲突
        unset PYTHONPATH
        unset LD_LIBRARY_PATH
        """
        # forwardGeometry(或 updateFramePlacements) 通常可以帮助更新 frame 的位姿
        # pin.updateFramePlacements(self.robot.model, self.robot.data)

        return q

    def get_link_quaternion(self, link_name=""):
        self._link_id = self.robot.model.getFrameId(link_name)
        _rot_world: np.ndarray = self.robot.data.oMf[self._link_id].rotation
        return R.from_matrix(_rot_world).as_quat(scalar_first=True)


class simulator:
    policy: ort.InferenceSession

    def __init__(self):
        # Load robot model
        self.spec = mujoco.MjSpec.from_file(cfg.mjcf_path)
        self._rehandle_xml()
        self.m = self.spec.compile()
        # self.m = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
        self.d = mujoco.MjData(self.m)
        self._scene = mujoco.MjvScene(self.m, 100000)
        print(f"Number of actuators: {self.m.nu}")

        self.m.opt.timestep = cfg.simulator_dt
        self.paused = False
        self._init_robot_conf()
        self._init_policy_conf()
        self.data_queue = queue.Queue()
        self.change_id = 0
        self.video_recorder = VideoRecorder(
            path=current_path + "/deploy_mujoco/recordings",
            tag=None,
            video_name="video_0",
            fps=int(1 / cfg.policy_dt),
            compress=False,
        )
        self.data_save = []
        with open("/home/hpx/HPX_LOCO_2/whole_body_tracking/my_variable.pkl", "rb") as f:
            self.data_collection = pickle.load(f)
            self.data_collection_index = 0
    def motion_play_isaac_sim(self):
        data = self.data_collection[self.data_collection_index]
        self.data_collection_index+=1
        if self.data_collection_index>=len(self.data_collection):
            self.data_collection_index = 0
        actor_obs = data["actor_obs"]
        command = data["command"]
        motion_ref_pos_b = data["motion_ref_pos_b"]
        motion_ref_ori_b = data["motion_ref_ori_b"]
        body_pos = data["body_pos"]
        body_ori = data["body_ori"]
        base_lin_vel = data["base_lin_vel"]
        base_ang_vel = data["base_ang_vel"]
        joint_pos = data["joint_pos"]
        joint_vel = data["joint_vel"]
        last_actions = data["last_actions"]
        new_actions = data["new_actions"]
        root_link_ang_vel_b = data["root_link_ang_vel_b"]
        root_link_lin_vel_w = data["root_link_lin_vel_w"]
        root_link_pos_w = data["root_link_pos_w"]
        root_link_quat_w = data["root_link_quat_w"]
        timesteps = np.array([data["timesteps"].astype(np.float32)])

        self.d.qpos[0:3] = root_link_pos_w
        self.d.qpos[3:7] = root_link_quat_w
        self.d.qpos[7 :] = joint_pos[self.isaac_sim2mujoco_index]
        self.d.qvel[0:3] = root_link_lin_vel_w
        self.d.qvel[3:6] = root_link_ang_vel_b
        self.d.qvel[6 :] = joint_vel[self.isaac_sim2mujoco_index]
        mujoco.mj_forward(self.m, self.d) 
        self.update_obs(timesteps)
        # self.obs = actor_obs
        self._policy_reasoning()
        return timesteps
    
    def motion_play(self):
        self.d.qpos[0:3] = (
            self.motion.body_pos_w[self.time_step, 0, :].detach().cpu().numpy()
        )
        q = self.motion.body_quat_w[self.time_step, 0, :].detach().cpu().numpy()[0, :]
        self.d.qpos[3:7] = q
        self.d.qpos[7 : 7 + len(self.default_pos)] = (
            self.motion.joint_pos[self.time_step].detach().cpu().numpy()
        )[:, self.isaac_sim2mujoco_index]
        self.d.qvel[0:3] = (
            self.motion.body_lin_vel_w[self.time_step, 0, :].detach().cpu().numpy()
        )
        self.d.qvel[3:6] = (
            self.motion.body_ang_vel_w[self.time_step, 0, :].detach().cpu().numpy()
        )
        self.d.qvel[6 : 6 + len(self.default_pos)] = (
            self.motion.joint_vel[self.time_step]
            .detach()
            .cpu()
            .numpy()[:, self.isaac_sim2mujoco_index]
        )
        mujoco.mj_forward(self.m, self.d)

    def run(self):
        save_data_flag = 1
        self.counter = 0
        self.d.qpos[7 : 7 + len(self.default_pos)] = self.default_pos
        self.d.qpos[2] = 0.992
        mujoco.mj_forward(self.m, self.d)
        self.target_dof_pos = self.default_pos.copy()[: self.action_num]
        self.phase = 0
        # self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        if save_data_flag:
            i = 0
            if os.path.exists("data.csv"):
                os.remove("data.csv")
        self.viewer = mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.key_callback
        )
        self.renderer = mujoco.renderer.Renderer(self.m, height=480, width=640)
        self.init_vel_geom(
            "Goal Vel: x: {:.2f}, y: {:.2f}, yaw: {:.2f},force_z:{:.2f}".format(
                self.cmd[0], self.cmd[1], self.cmd[2], 0.0
            )
        )
        self.prev_qpos = self.d.qpos
        # plot_thread = threading.Thread(target=self.plot_data, args=(self.data_queue,))
        # plot_thread.daemon = True
        # plot_thread.start()

        first_flag = False

        log = {
            "fps": [50],
            "dof_names": [joint.name for joint in self.spec.joints][1:],
            "body_names": self.mujoco_all_body_names,
            "dof_positions": [],
            "dof_velocities": [],
            "dof_torque": [],
            "body_positions": [],
            "body_rotations": [],
            "body_linear_velocities": [],
            "body_angular_velocities": [],
            "qpos": [],
            "qvel": [],
            "xpos": [],
            "xquat": [],
            "cvel": [],
            "P_gain": [self.P_gains],
            "D_gain": [self.D_gains],
            "target_pos": [],
        }

        while self.viewer.is_running():
            if not first_flag:
                first_flag = True
                if cfg.motion_play:
                    self.motion_play()
                    self.time_step *= 0
                    if cfg.sim_motion_play:
                        self.time_step[:] = self.motion_play_isaac_sim() * 1.0

                else:
                    self.motion_play()
                mujoco.mj_step(self.m, self.d)
                self.viewer.sync()
            self.policy_loop()
            # print(self.time_step, self.motion.time_step_total)
            log["dof_positions"].append(np.copy(self.d.qpos[7:]))
            log["dof_velocities"].append(np.copy(self.d.qvel[6:]))
            log["dof_torque"].append(np.copy(self.d.qfrc_actuator[6:]))
            log["body_positions"].append(
                np.copy(self.d.xpos[self.mujoco_body_names_indices, :])
            )
            log["body_rotations"].append(
                np.copy(self.d.xquat[self.mujoco_body_names_indices, :])
            )
            log["body_linear_velocities"].append(
                np.copy(self.d.cvel[self.mujoco_body_names_indices, 0:3])
            )
            log["body_angular_velocities"].append(
                np.copy(self.d.cvel[self.mujoco_body_names_indices, 3:6])
            )
            log["qpos"].append(np.copy(self.d.qpos))
            log["qvel"].append(np.copy(self.d.qvel))
            log["xpos"].append(
                np.copy(self.d.xpos[self.mujoco_body_names_indices, :])
            )
            log["xquat"].append(
                np.copy(self.d.xquat[self.mujoco_body_names_indices, :])
            )
            log["cvel"].append(
                np.copy(self.d.cvel[self.mujoco_body_names_indices, :])
            )
            log["target_pos"].append(np.copy(self.target_dof_pos))
            if self.time_step >= self.motion.time_step_total:
                break
        for k in (
            "dof_positions",
            "dof_velocities",
            "body_positions",
            "body_rotations",
            "body_linear_velocities",
            "body_angular_velocities",
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "cvel",
        ):
            log[k] = np.stack(log[k], axis=0)
        np.savez(
            "/home/hpx/HPX_LOCO_2/whole_body_tracking/deploy_mujoco/motion.npz", **log
        )
        # with open("data_save.pkl", "wb") as f:
        #     pickle.dump(self.data_save, f)
        print("stop")
        self.video_recorder.stop()

    def policy_loop(self):
        # print("="*(20))
        self.counter += 1
        # print(self.d.qvel[0])
        quat = self.d.qpos[3:7]
        omega = self.d.qvel[3:6]
        self.qpos = self.d.qpos[7 : ]
        self.P_n = self.qpos - self.default_pos
        self.V_n = self.d.qvel[6 : ]

        # if self.time_step >= 100:
        if self.time_step >= self.motion.time_step_total:
            self.time_step = 10

        if cfg.motion_play:
            if cfg.sim_motion_play:
                self.time_step[:] = self.motion_play_isaac_sim() * 1.0
            else:
                self.motion_play()
        else:
            self.update_obs(self.time_step)
            self.h2_action = self.h_action.copy()
            self.h_action = self.action.copy()
            self._policy_reasoning()
            # print(self.motion.joint_pos[self.time_step],"\r\n",self.r_joint_pos)
        action = (
            np.clip(
                copy.deepcopy(self.action[self.isaac_sim2mujoco_index]),
                -self.action_clip,
                self.action_clip,
            )
            * self.action_scale
            * self.tq_max
            / self.P_gains
            + self.default_pos
        )
        target_q = action.clip(-self.action_clip, self.action_clip)
        # print(target_q)
        self.target_dof_pos = target_q  # + self.default_pos[: self.action_num]
        self.time_step += 1
        self.contact_force()
        self.sim_loop()
        # mujoco.mjr_render(self._viewport, self._scene, self._context)
        # im = self.read_pixels()
        # self.video_recorder(im)
        # 更新 Renderer 场景，使用查看器的相机和选项，使图像与窗口一致
        self.renderer.update_scene(
            self.d,
            camera=self.viewer.cam,  # 使用查看器的相机视图
            scene_option=self.viewer.opt,  # 使用查看器的渲染选项
        )

        # 捕获图像：返回 (height, width, 3) 的 uint8 NumPy 数组 (RGB)
        img = self.renderer.render()
        self.video_recorder(img)

        self.viewer.sync()
        self.update_vel_geom()

    def update_obs(self, time_step):
        """
            +----------------------------------------------------------+
            | Active Observation Terms in Group: 'policy' (shape: (154,)) |
            +------------+--------------------------------+------------+
            |   Index    | Name                           |   Shape    |
            +------------+--------------------------------+------------+
            |     0      | command                        |   (58,)    |
            |     1      | motion_ref_ori_b               |    (6,)    |
            |     2      | base_ang_vel                   |    (3,)    |
            |     3      | joint_pos                      |   (29,)    |
            |     4      | joint_vel                      |   (29,)    |
            |     5      | actions                        |   (29,)    |
            +------------+--------------------------------+------------+
        """
        #################
        # motion joint pos command 27
        #################
        self.single_obs[:29] = np.copy(self.motion.joint_pos[time_step])
        #################
        # motion joint vel command 27
        #################
        self.single_obs[29:58] = np.copy(self.motion.joint_vel[time_step])

        self.robot_ref_quat_w = torch.from_numpy(self.d.qpos[3:7]).unsqueeze(0)  # shape [n,4]
        self.ref_quat_w = self.motion.body_quat_w[time_step, 7, :]  # shape [n,4]

        q01 = self.robot_ref_quat_w
        q02 = self.ref_quat_w
        q10 = quat_inv(q01)
        if q02 is not None:
            q12 = quat_mul(q10, q02)
        else:
            q12 = q10
        mat = matrix_from_quat(q12)
        motion_ref_ori_b = mat[..., :2].reshape(mat.shape[0], -1)  # shape [n,6]
        #################
        # motion_ref_ori_b 6
        #################
        self.single_obs[58:64] = np.copy(motion_ref_ori_b)
        #################
        # base_ang_vel 3
        #################
        self.single_obs[64:67] = self.d.qvel[3:6]
        #################
        # joint_pos 29
        #################
        self.single_obs[67:96] = (
            self.d.qpos[7:]# - self.default_pos
        )[self.mujoco2isaac_sim_index]
        #[0, 7, 27, 14, 1, 8, 28, 15, 21, 2, 9, 16, 22, 3, 10, 17, 23, 4, 11, 18, 24, 5, 12, 19, 25, 6, 13, 20, 26]
        #################
        # joint_vel 29
        #################
        self.single_obs[96:125] = self.d.qvel[6:][self.mujoco2isaac_sim_index]
        #################
        # actions 29
        #################
        self.single_obs[125:154] = self.action  # / self.action_scale

        self.obs = (
            torch.tensor(np.concatenate([self.single_obs] * cfg.frame_stack, axis=-1))
            .clamp(-10, 10)
            .unsqueeze(0)
            .detach()
            .cpu()
            .numpy()
        )

    def _policy_reasoning(self):

        if cfg.policy_type == "onnx":
            (
                act,
                self.r_joint_pos,
                self.r_joint_vel,
                self.r_body_pos_w,
                self.r_body_quat_w,
                self.r_body_lin_vel_w,
                self.r_body_ang_vel_w,
            ) = self.run_onnx_inference(
                self.policy, self.obs.astype(np.float32), self.time_step
            )
        self.action[:] = act.copy()

    def sim_loop(self):
        for i in range(self.control_decimation):
            step_start = time.time()

            if not cfg.motion_play or (cfg.motion_play and cfg.sim_motion_play):
                tau = self._PD_control(self.target_dof_pos)
                self.d.ctrl[:] = tau*0
            if not self.paused:
                self.prev_qpos = self.d.qpos.copy()
                self.set_camera()

                mujoco.mj_step(self.m, self.d)

            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def _PD_control(self, _P_t=0):
        P_n = self.d.qpos[7:]
        V_n = self.d.qvel[6:]

        KP = self.P_gains
        KD = self.D_gains
        # 在_compute_torques中使用
        t = KP * (_P_t - P_n) - KD * V_n
        return t

    def contact_force(self):
        force = 0
        for contact_id, contact in enumerate(self.d.contact):
            if contact.efc_address >= 0:  # Valid contact
                forcetorque = np.zeros(6)
                mujoco.mj_contactForce(self.m, self.d, contact_id, forcetorque)
                # print("forcetorque: ",forcetorque)
                force += forcetorque[0]
        self.fz = force / 65 / 9.81
        # print("force: %8.3f"% force)

    def key_callback(self, keycode):
        # 按空格键切换暂停/继续

        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'running'}")
        elif chr(keycode).lower() == "w":
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0
            self.cmd[0] = 0.8
        elif chr(keycode).lower() == "s":
            self.cmd[0] = -0.8
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

        elif chr(keycode).lower() == "a":
            self.cmd[1] = 0.4
            self.cmd[0] = 0.0
            self.cmd[2] = 0.0
        elif chr(keycode).lower() == "d":
            self.cmd[1] = -0.4
            self.cmd[0] = 0.0
            self.cmd[2] = 0.0
        elif chr(keycode).lower() == "q":
            self.cmd[2] = 1.5
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
        elif chr(keycode).lower() == "e":
            self.cmd[2] = -1.5
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
        # 释放键时重置控制量
        elif keycode == 48:  # keycode=0 表示无按键
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

    def set_camera(self):
        self.viewer.cam.distance = 4
        self.viewer.cam.azimuth = 180  # 135
        self.viewer.cam.elevation = 0.0
        self.viewer.cam.fixedcamid = -1
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = 1

    def _init_robot_conf(self):
        self.default_pos = np.array(
            cfg.arm_default_pos+ cfg.pelvis_default_pos + cfg.leg_default_pos  + cfg.head_default_pos,
            dtype=np.float32,
        )
        self.P_gains = np.array(
            cfg.arm_P_gains+ cfg.pelvis_P_gains + cfg.leg_P_gains  + cfg.head_P_gains,
        )
        self.D_gains = np.array(
            cfg.arm_D_gains+ cfg.pelvis_D_gains + cfg.leg_D_gains  + cfg.head_D_gains,
        )
        self.P_n = np.zeros_like(self.default_pos)
        self.V_n = np.zeros_like(self.default_pos)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.isaac_sim2mujoco_index = [
            cfg.isaac_sim_joint_name.index(name) for name in cfg.mujoco_joint_name
        ]
        self.mujoco2isaac_sim_index = [
            cfg.mujoco_joint_name.index(name) for name in cfg.isaac_sim_joint_name
        ]
        self.mujoco_all_body_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.m.nbody)
        ][1:]
        self.mujoco_body_names_indices = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.mujoco_all_body_names
        ]
        self.motion_body_names_in_isaacsim_index = [
            cfg.isaac_sim_link_name.index(name) for name in cfg.motion_body_names
        ]
        a = 1

    def _init_policy_conf(self):
        self.body_indexes = torch.tensor(
            self.motion_body_names_in_isaacsim_index,
            dtype=torch.long,
            device="cpu",
        )
        self.motion = MotionLoader(
            cfg.motion_file,
            self.body_indexes,
            "cpu",
        )
        self.policy_dt = cfg.policy_dt
        if cfg.motion_play:
            self.policy_dt = 1 / self.motion.fps
        self.control_decimation = int(self.policy_dt / cfg.simulator_dt)
        print("control_decimation: ", self.control_decimation)
        if cfg.policy_type == "torch":
            self.policy = torch.jit.load(cfg.policy_path)
        elif cfg.policy_type == "onnx":
            self.policy = self.load_onnx_model(cfg.policy_path)

        self.h2_action = np.zeros(cfg.action_num, dtype=np.float32)
        self.h_action = np.zeros(cfg.action_num, dtype=np.float32)
        self.action = np.zeros(cfg.action_num, dtype=np.float32)
        self.action_clip = cfg.action_clip

        self.tq_max = np.array(
            cfg.arm_tq_max + cfg.pelvis_tq_max + cfg.leg_tq_max + cfg.head_tq_max,
            dtype=np.float32,
        )
        self.action_scale = cfg.action_scale
        self.action_num = cfg.action_num
        self.obs = np.zeros(cfg.num_single_obs * cfg.frame_stack, dtype=np.float32)
        self.time_step = np.ones(1, dtype=np.float32) * 1
        self.single_obs = np.zeros(cfg.num_single_obs, dtype=np.float32)

    def load_onnx_model(self, onnx_path, device="cpu"):
        providers = (
            ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        session = ort.InferenceSession(onnx_path, providers=providers)
        return session

    def run_onnx_inference(self, session, obs, time_step):
        # 转换为numpy array并确保数据类型正确
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if isinstance(time_step, torch.Tensor):
            time_step = time_step.detach().cpu().numpy()
        # 获取输入名称
        obs_name = session.get_inputs()[0].name
        time_step_name = session.get_inputs()[1].name

        # 运行推理
        (
            actions,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = session.run(
            None,
            {
                obs_name: obs.reshape(1, cfg.num_single_obs),
                time_step_name: time_step.reshape(1, 1),
            },
        )
        return (
            actions,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        )  # 默认返回第一个输出

    def _rehandle_xml(self):

        joints_to_remove, actuators_to_remove, _ = self._get_spec_modifications(
            only_leg=cfg.only_leg_flag, with_wrist=cfg.with_wrist_flag
        )
        for actuator in self.spec.actuators:
            if actuator.name in actuators_to_remove:
                actuator.delete()
        for joint in self.spec.joints:
            if joint.name in joints_to_remove:
                joint.delete()

    def _get_spec_modifications(
        self, only_leg, with_wrist
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, actuators, and equality constraints should be removed from the Mujoco specification.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing lists of joints to remove, actuators to remove,
            and equality constraints to remove.
        """

        joints_to_remove = [
            # Left Hand
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            # Right Hand
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
        ]

        actuators_to_remove = [
            # Left Hand
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            # Right Hand
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
        ]
        if not with_wrist:
            joints_to_remove += [
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
            actuators_to_remove += [
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
        if only_leg:
            joints_to_remove += [
                # Left Arm
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                # Right Arm
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "torso_joint",
            ]
            actuators_to_remove += [
                # Left Arm
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                # Right Arm
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "torso_joint",
            ]

        equ_constr_to_remove = []

        return joints_to_remove, actuators_to_remove, equ_constr_to_remove

    def plot_data(self, data_queue):
        print("plot_data")
        plt.ion()  # 开启交互模式
        first_flag = 1

        while True:
            if not data_queue.empty():
                merged_tensor = data_queue.get()
                plot_num = merged_tensor.shape[0]
                if first_flag:
                    first_flag = 0
                    # 计算行数和列数
                    rows = math.floor(math.sqrt(plot_num))
                    cols = math.ceil(plot_num / rows)

                    fig, axs = plt.subplots(rows, cols, figsize=(10, 12))  # 创建子图
                    axs = axs.flatten()  # 将二维数组展平成一维数组，方便索引

                    lines = [ax.plot([], [])[0] for ax in axs]  # 初始化每个子图的线条
                    xdata = [
                        [0 for _ in range(700)] for _ in range(plot_num)
                    ]  # 存储每个子图的 x 数据
                    ydata = [
                        [0] * 700 for _ in range(plot_num)
                    ]  # 存储每个子图的 y 数据

                    from matplotlib.widgets import Slider

                    # Add slider
                    ax_slider = plt.axes([0.15, 0.02, 0.65, 0.03])  # Slider position
                    self.slider = Slider(
                        ax_slider, "Control", 0.1, 3.0, valinit=1.0, valstep=0.001
                    )
                    self.slider.on_changed(self.update_sld)
                for i in range(plot_num):
                    xdata[i].append(len(xdata[i]))
                    ydata[i].append(merged_tensor[i].item())
                    lines[i].set_data(xdata[i][-100:], ydata[i][-100:])
                    axs[i].relim()
                    axs[i].autoscale_view()
                # print(len(xdata[i]))
                if len(xdata[i]) % 1 == 0:
                    fig.canvas.draw()
                    fig.canvas.flush_events()

    def update_sld(self, val):
        slider_value = self.slider.val  # Get slider value
        self.D_gains[self.change_id] = slider_value * self.D_gains[self.change_id + 6]
        print(f"D_gains {self.change_id:d} value: {self.D_gains[self.change_id]:.2f}")

    def init_vel_geom(self, input):
        # create an invisibale geom and add label on it
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LABEL,
            size=np.array([0.2, 0.2, 0.2]),  # label_size
            pos=self.d.qpos[:3]
            + np.array(
                [0.0, 0.0, 1.0]
            ),  # lebel position, here is 1 meter above the root joint
            mat=np.eye(3).flatten(),  # label orientation, here is no rotation
            rgba=np.array([0, 0, 0, 0]),  # invisible
        )
        geom.label = str(input)  # set label text
        self.viewer.user_scn.ngeom += 1

    def update_vel_geom(self):
        # update the geom position and label text
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1]
        geom.pos = self.d.qpos[:3] + np.array([0.0, 0.0, 1.0])
        geom.label = "rb h{:.2f} \r\nGoal Vel: x: {:.2f}, y: {:.2f}, yaw: {:.2f},force_z: {:.2f}".format(
            # self.data["robot.data.body_pos_w"].detach().cpu().numpy()[0][2],
            0.0,
            self.cmd[0],
            self.cmd[1],
            self.cmd[2],
            self.fz,
        )


if __name__ == "__main__":
    s = simulator()
    s.run()
