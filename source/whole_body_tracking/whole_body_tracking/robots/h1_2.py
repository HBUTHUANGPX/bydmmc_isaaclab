import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots import unitree_actuators
from whole_body_tracking.robots.unitree_actuators import (
    UnitreeActuatorCfg_M107_15,
    UnitreeActuatorCfg_M107_24,
    UnitreeActuatorCfg_N7520_14p3,
    UnitreeActuatorCfg_N7520_22p5,
    UnitreeActuatorCfg_N5020_16,
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

H1_2_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/unitree_h1_2/h1_2_handless.urdf",
        # asset_path=f"{ASSET_DIR}/unitree_h1_2/h1_2.urdf",
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
        pos=(0.0, 0.0, 1.03),
        joint_pos={
            ".*_hip_pitch_joint": -0.18,
            ".*_knee_joint": 0.38,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_joint": 1.54,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # actuators_old={
    #     "legs": ImplicitActuatorCfg(
    #         joint_names_expr=[
    #             ".*_hip_yaw_joint",
    #             ".*_hip_pitch_joint",
    #             ".*_hip_roll_joint",
    #             ".*_knee_joint",
    #         ],
    #         effort_limit_sim={
    #             ".*_hip_yaw_joint": 200.0,
    #             ".*_hip_pitch_joint": 200.0,
    #             ".*_hip_roll_joint": 200.0,
    #             ".*_knee_joint": 300.0,
    #         },
    #         velocity_limit_sim={
    #             ".*_hip_yaw_joint": 23.0,
    #             ".*_hip_pitch_joint": 23.0,
    #             ".*_hip_roll_joint": 23.0,
    #             ".*_knee_joint": 14.0,
    #         },
    #         stiffness={
    #             ".*_hip_yaw_joint": 200,
    #             ".*_hip_pitch_joint": 300,
    #             ".*_hip_roll_joint": 350,
    #             ".*_knee_joint": 300,
    #         },
    #         damping={
    #             ".*_hip_yaw_joint": 1.5,
    #             ".*_hip_pitch_joint": 4,
    #             ".*_hip_roll_joint": 4,
    #             ".*_knee_joint": 4,
    #         },
    #         armature={
    #             ".*_hip_yaw_joint": 0.01,
    #             ".*_hip_pitch_joint": 0.01,
    #             ".*_hip_roll_joint": 0.01,
    #             ".*_knee_joint": 0.015,
    #         },
    #     ),
    #     "feet": ImplicitActuatorCfg(
    #         effort_limit_sim=50.0,
    #         velocity_limit_sim=37.0,
    #         joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    #         stiffness=70.0,
    #         damping=2,
    #         armature=0.01,
    #     ),
    #     "torso": ImplicitActuatorCfg(
    #         effort_limit_sim=200,
    #         velocity_limit_sim=23.0,
    #         joint_names_expr=["torso_joint"],
    #         stiffness=200,
    #         damping=3.5,
    #         armature=0.01,
    #     ),
    #     "arms": ImplicitActuatorCfg(
    #         joint_names_expr=[
    #             ".*_shoulder_pitch_joint",
    #             ".*_shoulder_roll_joint",
    #             ".*_shoulder_yaw_joint",
    #             ".*_elbow_joint",
    #             ".*_wrist_roll_joint",
    #             ".*_wrist_pitch_joint",
    #             ".*_wrist_yaw_joint",
    #         ],
    #         effort_limit_sim={
    #             ".*_shoulder_pitch_joint": 40.0,
    #             ".*_shoulder_roll_joint": 40.0,
    #             ".*_shoulder_yaw_joint": 18.0,
    #             ".*_elbow_joint": 18.0,
    #             ".*_wrist_roll_joint": 19.0,
    #             ".*_wrist_pitch_joint": 19.0,
    #             ".*_wrist_yaw_joint": 19.0,
    #         },
    #         velocity_limit_sim={
    #             ".*_shoulder_pitch_joint": 9.0,
    #             ".*_shoulder_roll_joint": 9.0,
    #             ".*_shoulder_yaw_joint": 20.0,
    #             ".*_elbow_joint": 20.0,
    #             ".*_wrist_roll_joint": 31.4,
    #             ".*_wrist_pitch_joint": 31.4,
    #             ".*_wrist_yaw_joint": 31.4,
    #         },
    #         stiffness={
    #             ".*_shoulder_pitch_joint": 60,
    #             ".*_shoulder_roll_joint": 60,
    #             ".*_shoulder_yaw_joint": 20,
    #             ".*_elbow_joint": 20,
    #             ".*_wrist_roll_joint": 20,
    #             ".*_wrist_pitch_joint": 20,
    #             ".*_wrist_yaw_joint": 20,
    #         },
    #         damping={
    #             ".*_shoulder_pitch_joint": 1.5,
    #             ".*_shoulder_roll_joint": 1.5,
    #             ".*_shoulder_yaw_joint": 2,
    #             ".*_elbow_joint": 1,
    #             ".*_wrist_roll_joint": 0.8,
    #             ".*_wrist_pitch_joint": 0.8,
    #             ".*_wrist_yaw_joint": 0.8,
    #         },
    #         armature={
    #             ".*_shoulder_pitch_joint": 0.1,
    #             ".*_shoulder_roll_joint": 0.1,
    #             ".*_shoulder_yaw_joint": 0.1,
    #             ".*_elbow_joint": 0.1,
    #             ".*_wrist_roll_joint": 0.1,
    #             ".*_wrist_pitch_joint": 0.1,
    #             ".*_wrist_yaw_joint": 0.1,
    #         },
    #     ),
    # },
    actuators={
        "M107_15":UnitreeActuatorCfg_M107_15(
            joint_names_expr=[
                "torso_joint",
                ".*_hip_yaw_joint"
            ],
            stiffness=200,
            damping=1.5,
            friction=0.01,
            effort_limit_sim= 200,
        ),
        "M107_24":UnitreeActuatorCfg_M107_24(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_knee_joint",
            ],
            stiffness=300,
            damping=3.0,
            friction=0.01,
            effort_limit_sim= 300,
        ),
        "N7520_14p3":UnitreeActuatorCfg_N7520_14p3(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                ".*_shoulder_yaw_joint",
            ],
            stiffness=70,
            damping=2.0,
            friction=0.01,
            effort_limit_sim= 75,
        ),
        "N7520_22p5":UnitreeActuatorCfg_N7520_22p5(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_joint",
            ],
            stiffness=70,
            damping=1.5,
            friction=0.01,
            effort_limit_sim= 120,
        ),
        "N5020_16":UnitreeActuatorCfg_N5020_16(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_yaw_joint",
            ],
            stiffness=20,
            damping=0.8,
            friction=0.01,
            effort_limit_sim= 25,
        )
    }
)

H1_2_ACTION_SCALE = {}
for a in H1_2_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            H1_2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
