"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--motion_file", type=str, default=None, help="Path to the motion file."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        # art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        # if art is None:
        #     print("[WARN] No model artifact found in the run.")
        # else:
        #     env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    print("export_model_dir:", export_model_dir)
    attach_onnx_metadata(
        env.unwrapped,
        args_cli.wandb_path if args_cli.wandb_path else "none",
        export_model_dir,
    )
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    log = []
    save_flag=True
    save_step = int(50*20)
    while simulation_app.is_running() and save_step:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            if save_flag:
                save_step-=1
                if (save_step%50)==0:
                    print(save_step)
                c_obs = env.unwrapped.observation_manager.compute()["critic"]
                actor_obs = obs[0, :]
                command = c_obs[0, :58]
                motion_ref_pos_b = c_obs[0, 58:61]
                motion_ref_ori_b = c_obs[0, 61:67]
                body_pos = c_obs[0, 67:112]
                body_ori = c_obs[0, 112:202]
                base_lin_vel = c_obs[0, 202:205]
                base_ang_vel = c_obs[0, 205:208]
                joint_pos = c_obs[0, 208:237]
                joint_vel = c_obs[0, 237:266]
                last_actions = c_obs[0, 266:295]
                new_actions = actions[0, :]
                root_link_ang_vel_b = env.unwrapped.scene["robot"].data.root_link_ang_vel_b[
                    0, :
                ]
                root_link_lin_vel_w = env.unwrapped.scene["robot"].data.root_link_lin_vel_w[
                    0, :
                ]
                root_link_pos_w = env.unwrapped.scene["robot"].data.root_link_pos_w[0, :]
                root_link_quat_w = env.unwrapped.scene["robot"].data.root_link_quat_w[0, :]
                timesteps = env.unwrapped.command_manager.get_term("motion").time_steps[0]
                # <whole_body_tracking.tasks.tracking_q1.mdp.commands.MotionCommand object at 0x7f42e7381540>
                _log = {
                    "actor_obs":actor_obs.detach().cpu().numpy(),
                    "command": command.detach().cpu().numpy(),
                    "motion_ref_pos_b": motion_ref_pos_b.detach().cpu().numpy(),
                    "motion_ref_ori_b": motion_ref_ori_b.detach().cpu().numpy(),
                    "body_pos": body_pos.detach().cpu().numpy(),
                    "body_ori": body_ori.detach().cpu().numpy(),
                    "base_lin_vel": base_lin_vel.detach().cpu().numpy(),
                    "base_ang_vel": base_ang_vel.detach().cpu().numpy(),
                    "joint_pos": joint_pos.detach().cpu().numpy(),
                    "joint_vel": joint_vel.detach().cpu().numpy(),
                    "last_actions": last_actions.detach().cpu().numpy(),
                    "new_actions": new_actions.detach().cpu().numpy(),
                    "root_link_ang_vel_b": root_link_ang_vel_b.detach().cpu().numpy(),
                    "root_link_lin_vel_w": root_link_lin_vel_w.detach().cpu().numpy(),
                    "root_link_pos_w": root_link_pos_w.detach().cpu().numpy(),
                    "root_link_quat_w": root_link_quat_w.detach().cpu().numpy(),
                    "timesteps": timesteps.detach().cpu().numpy(),
                }
                log.append(_log)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    if save_flag:
        import pickle
        file_path = 'my_variable.pkl'  # 文件扩展名通常为 .pkl 或 .pickle
        with open(file_path, 'wb') as f:
            pickle.dump(log, f)            
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
