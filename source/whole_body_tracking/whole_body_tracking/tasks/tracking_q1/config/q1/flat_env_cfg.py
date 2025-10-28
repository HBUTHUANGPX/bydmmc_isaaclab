from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking_q1.tracking_env_cfg import TrackingEnvCfg

from whole_body_tracking.robots.q1 import Q1_ACTION_SCALE, Q1_CYLINDER_CFG

@configclass
class Q1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Q1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = Q1_ACTION_SCALE
        self.commands.motion.reference_body = "torso_link"
        self.commands.motion.body_names = [
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

