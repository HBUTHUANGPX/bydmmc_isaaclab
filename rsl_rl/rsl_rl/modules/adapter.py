from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.utils import resolve_nn_activation


# class HistortEncoder(nn.Module):
#     def __init__(
#         self,
#         history_obs_dim: int = 10350,
#         history_hidden_dims: list[int] = [256, 256, 256],
#         activation: str = "elu",
#     ):
#         super().__init__()
#         self.activation = resolve_nn_activation(activation)

#         # History Encoder: 输入历史数据 (batch, 10350) -> 输出 e_t (batch, 512)
#         history_layers = []
#         self.history_obs_dim = history_obs_dim

#         prev_dim = history_obs_dim
#         for dim in history_hidden_dims:
#             history_layers.append(nn.Linear(prev_dim, dim))
#             history_layers.append(self.activation)
#             prev_dim = dim
#         history_layers.append(nn.Linear(prev_dim, 512))
#         self.history_encoder = nn.Sequential(*history_layers)


#     def forward(self,obs: torch.Tensor):
#         history = obs[:, : self.history_obs_dim]
#         return
class HistoryWorldModel(nn.Module):
    def __init__(
        self,
        history_input_dim: int = 84 * 4,
        history_hidden_dims: list[int] = [256, 256, 256],
        world_hidden_dims: list[int] = [256, 256, 256],
        obs_single_frame_dim: int = 144,
        next_obs_predict_frame_dim: int = 84 + 3 + 27 + 27 + 27,
        activation: str = "elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        latent_space=32,
        est_space=132,
    ):
        super().__init__()
        self.activation = resolve_nn_activation(activation)

        # History Encoder: 输入历史数据 (batch, 10350) -> 输出 e_t (batch, 512)
        self.latent_space = latent_space
        self.est_space = est_space
        history_layers = []
        self.history_obs_dim = obs_single_frame_dim * 4
        self.history_encoder_input_dim = history_input_dim
        self.next_obs_predict_frame_dim = next_obs_predict_frame_dim
        self.start_offsets = [144 * i + 60 for i in range(4)]
        prev_dim = self.history_encoder_input_dim
        for dim in history_hidden_dims:
            history_layers.append(nn.Linear(prev_dim, dim))
            history_layers.append(self.activation)
            prev_dim = dim
        # history_layers.append(nn.Linear(prev_dim, 16 + 132))
        self.history_encoder = nn.Sequential(*history_layers)
        self.encode_mean_latent = nn.Linear(prev_dim, self.latent_space)
        self.encode_logvar_latent = nn.Linear(prev_dim, self.latent_space)
        self.encode_mean_vel = nn.Linear(prev_dim, self.est_space)
        self.encode_logvar_vel = nn.Linear(prev_dim, self.est_space)
        # 添加LayerNorm用于latent space归一化
        self.latent_norm = nn.LayerNorm(self.latent_space, elementwise_affine=False)

        # World Model: 输入 e_t + 当前帧 (batch, 662) -> 输出未来 20 帧 (batch, 3000)
        world_input_dim = self.latent_space + self.est_space
        world_layers = []
        prev_dim = world_input_dim
        for dim in world_hidden_dims:
            world_layers.append(nn.Linear(prev_dim, dim))
            world_layers.append(self.activation)
            prev_dim = dim
        world_layers.append(nn.Linear(prev_dim, next_obs_predict_frame_dim))
        self.world_model = nn.Sequential(*world_layers)

        print(f"history_encoder MLP: {self.history_encoder}")
        print(f"world_model MLP: {self.world_model}")

    def reparameterise(self, mean, logvar):
        var = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(var)
        code = mean + var * code_temp
        return code

    def cenet_forward(self, obs_history):

        distribution = self.history_encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        code_vel = self.reparameterise(mean_vel, logvar_vel)
        # 对latent space进行归一化
        normalized_code_latent = self.latent_norm(code_latent) # * 1e-4
        code = torch.cat((code_vel, normalized_code_latent), dim=-1)
        decode = self.world_model(code)
        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    def forward(self, obs):
        history = obs[:, : self.history_obs_dim]
        # slices = [history[:, start : start + 84] for start in self.start_offsets]
        # result = torch.cat(slices, dim=1)
        code, _, _, _, _, _, _ = self.cenet_forward(history)
        return code

    def update(self, obs: torch.Tensor, next_privilige: torch.Tensor, beta=1):
        history = obs[:, : self.history_obs_dim]
        # slices = [history[:, start : start + 84] for start in self.start_offsets]
        # result = torch.cat(slices, dim=1)
        code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent = (
            self.cenet_forward(history)
        )
        # encoder loss
        encoder_loss = nn.functional.mse_loss(
            code_vel, next_privilige[:, 54 : 54 + self.est_space]
        )
        # beta-VAE loss
        beta_VAE = beta * (
            -0.5
            * torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())
        )
        # state predict loss
        wm_loss = nn.functional.mse_loss(
            decode, next_privilige
            # decode, next_privilige[:, -self.next_obs_predict_frame_dim:]
        )
        return encoder_loss, beta_VAE, wm_loss
