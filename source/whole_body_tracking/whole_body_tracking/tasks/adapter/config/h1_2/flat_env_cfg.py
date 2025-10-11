from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from whole_body_tracking.tasks.adapter.adapter_env_cfg import AdapterEnvCfg
from whole_body_tracking.robots.h1_2 import H1_2_ACTION_SCALE, H1_2_CYLINDER_CFG

@configclass
class H1_2FlatEnvCfg(AdapterEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = H1_2_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = H1_2_ACTION_SCALE
        self.commands.motion.reference_body = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",

            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",

            "torso_link",

            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

