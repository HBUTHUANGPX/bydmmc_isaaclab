# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.modules import HistoryWorldModel
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
from rsl_rl.utils import resolve_nn_activation


class ActorCriticAdapter(nn.Module):
    is_recurrent = False
    is_adapter = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        history_encoder_hidden_dims: list[int] = [512, 512],
        world_model_hidden_dims: list[int] = [512, 512],
        estimator_future_frame: int = 20,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        frozen_actor_path: str = "",  # 冻结的网络 A
        frame_dim: int = 144,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.adapter_flag = True  # False， True
        # 激活函数（硬编码 elu）
        self.activation = resolve_nn_activation(activation)
        self.current_frame_dim = 54+6+3+27+27+27  # 144
        if self.adapter_flag:
            # History 和 World Model 模块
            history_input_dim = 4 * (54+6+3+27+27+27)  # 69*144
            # history_input_dim = 4 * (3+27+27+27)  # 69*144
            self.latent_space = 3
            self.est_space = 3
            self.history_world_model = HistoryWorldModel(
                history_input_dim=history_input_dim,
                history_hidden_dims=history_encoder_hidden_dims,
                world_hidden_dims=world_model_hidden_dims,
                obs_single_frame_dim=frame_dim,
                next_obs_predict_frame_dim=num_critic_obs,
                # next_obs_predict_frame_dim=84 + 3 + 27 + 27 + 27,
                activation=activation,
                latent_space=self.latent_space,
                est_space=self.est_space,
            )

            # 网络 B（输入 e_t 512，可学习）
            actor_layers = []
            actor_layers.append(
                nn.Linear(
                    self.latent_space + self.est_space + self.current_frame_dim,
                    actor_hidden_dims[0],
                )
            )
            actor_layers.append(self.activation)
            for i in range(len(actor_hidden_dims) - 1):
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1])
                )
                actor_layers.append(self.activation)
            actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
            self.actor = nn.Sequential(*actor_layers)
        else:
            actor_layers = []
            actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
            actor_layers.append(self.activation)
            for i in range(len(actor_hidden_dims) - 1):
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1])
                )
                actor_layers.append(self.activation)
            actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
            self.actor = nn.Sequential(*actor_layers)

        # Critic（不变）
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(self.activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(
                        critic_hidden_dims[layer_index],
                        critic_hidden_dims[layer_index + 1],
                    )
                )
                critic_layers.append(self.activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor B MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # 动作噪声
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

        self.distribution: Normal = None
        # 禁用分布验证以加速
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        if self.adapter_flag:
            current = observations[:, -self.current_frame_dim :]  # 当前帧
            with torch.no_grad():
                e_t = self.history_world_model(observations)
            mean = self.actor(torch.cat([e_t, current], dim=1))
        else:
            mean = self.actor(observations)

        # 计算标准差
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)

        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        self.update_distribution(observations)
        return self.distribution.mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
