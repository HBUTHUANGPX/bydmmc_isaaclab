import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots import unitree_actuators
from whole_body_tracking.robots.unitree_actuators import (
    EncosActuatorCfg_EC_A8112,
    EncosActuatorCfg_EC_A6408,
    EncosActuatorCfg_EC_A10020,
    EncosActuatorCfg_EC_A4310,
    Ti5ActuatorCfg_CRA_RI60_80,
    Ti5ActuatorCfg_CRA_RI50_70,
    Ti5ActuatorCfg_CRA_RI40_52,
    Ti5ActuatorCfg_CRA_RI30_40,
    HTActuatorCfg_DMS_6015,
    HTActuatorCfg_DMS_6015_2
)

Q1_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/Q1P01/urdf/Q1P01_urdf_rl.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.992),
        joint_pos={
            ".*_hip_pitch_joint": -0.18,
            ".*_knee_joint": 0.38,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_joint": 0.0,
            "L_shoulder_roll_joint": 0.2,
            "L_shoulder_pitch_joint": 0.0,
            "R_shoulder_roll_joint": -0.2,
            "R_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "EC_A8112":EncosActuatorCfg_EC_A8112(
            joint_names_expr=[
                ".*_hip_roll_joint"
            ],
            stiffness=90,
            damping=1.5,
            friction=0.01,
            effort_limit_sim= 90,
        ),
        "EC_A6408":EncosActuatorCfg_EC_A6408(
            joint_names_expr=[
                ".*_hip_yaw_joint"
            ],
            stiffness=60,
            damping=1.5,
            friction=0.01,
            effort_limit_sim= 60,
        ),
        "EC_A10020":EncosActuatorCfg_EC_A10020(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            stiffness=330,
            damping=1.5,
            friction=0.01,
            effort_limit_sim= 330,
        ),
        "EC_A4310":EncosActuatorCfg_EC_A4310(
            joint_names_expr=[
                ".*_ankle_roll_joint",
                ".*_ankle_pitch_joint",
            ],
            stiffness=36,
            damping=1.5,
            friction=0.01,
            effort_limit_sim= 36,
        ),

        "CRA_RI60_80":Ti5ActuatorCfg_CRA_RI60_80(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                "pelvis_joint",
            ],
            stiffness=42,
            damping=1.0,
            friction=0.01,
            effort_limit_sim= 42,
        ),

        "CRA_RI50_70":Ti5ActuatorCfg_CRA_RI50_70(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            stiffness=23,
            damping=1.0,
            friction=0.01,
            effort_limit_sim= 23,
        ),

        "CRA_RI40_52":Ti5ActuatorCfg_CRA_RI40_52(
            joint_names_expr=[
                ".*_forearm_yaw_joint",
            ],
            stiffness=8,
            damping=1.0,
            friction=0.01,
            effort_limit_sim= 8.3,
        ),

        "CRA_RI30_40":Ti5ActuatorCfg_CRA_RI30_40(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_yaw_joint",
            ],
            stiffness=8,
            damping=1.0,
            friction=0.01,
            effort_limit_sim= 8.3,
        ),

        "HT_DMS_6015_2":HTActuatorCfg_DMS_6015_2(
            joint_names_expr=[
                "head_yaw_joint",
            ],
            stiffness=3.0,
            damping=0.6,
            friction=0.01,
            effort_limit_sim= 1.26*2,
        ),
        "HT_DMS_6015":HTActuatorCfg_DMS_6015(
            joint_names_expr=[
                "head_pitch_joint",
            ],
            stiffness=1.5,
            damping=0.3,
            friction=0.01,
            effort_limit_sim= 1.26,
        ),
    }
)

Q1_ACTION_SCALE = {}
for a in Q1_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            Q1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
