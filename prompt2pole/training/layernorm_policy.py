"""SAC policy with LayerNorm in the critic and actor MLPs (D-0027).

EXP-0015 (reward v4) reproduced the textbook **SAC Q-value divergence/collapse** on the stock
``MlpPolicy``: an unnormalized ReLU-MLP critic ([256, 256]) extrapolates without bound off the
data, ``critic_loss`` climbed 3.4 -> 10 while the policy's Q dropped +10 -> -11 and the early
peak (ep_rew +14) collapsed to -14 (catastrophic forgetting). Inserting **LayerNorm after each
linear layer** (Linear -> LayerNorm -> activation) bounds the critic's outputs, makes value
iteration non-expansive, and mitigates overestimation / plasticity loss — the architectural fix
for exactly this failure (SEEM arXiv:2310.04411; CrossQ arXiv:1902.05605; SimBa arXiv:2410.09754;
BRO arXiv:2405.16158). LayerNorm in **both** critic and actor beats either alone (SimBa).

This keeps the SAC *algorithm* untouched (no target-net removal, no BatchNorm train/eval coupling);
only the MLP architecture changes — a custom network on the stock SB3 SAC. Amends D-0004; gated and
selected only when ``critic_layernorm`` is enabled (default stays the stock MlpPolicy).
"""

from __future__ import annotations

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import Actor, SACPolicy
from torch import nn

# Inserted by create_mlp AFTER each hidden Linear and BEFORE its activation (pre-nonlinearity
# LayerNorm; not applied to the output layer, so Q / mu / log_std heads stay un-normalized).
_NORM = [nn.LayerNorm]


class LayerNormActor(Actor):
    """``Actor`` whose ``latent_pi`` MLP has LayerNorm after each hidden linear layer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Rebuild latent_pi with LayerNorm (same dims -> mu/log_std heads remain compatible).
        self.latent_pi = nn.Sequential(
            *create_mlp(
                self.features_dim, -1, self.net_arch, self.activation_fn,
                post_linear_modules=_NORM,
            )
        )


class LayerNormContinuousCritic(ContinuousCritic):
    """``ContinuousCritic`` whose Q-network MLPs have LayerNorm after each hidden linear layer."""

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        activation_fn=nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ) -> None:
        super().__init__(
            observation_space, action_space, net_arch, features_extractor, features_dim,
            activation_fn, normalize_images, n_critics, share_features_extractor,
        )
        # Rebuild each qf with LayerNorm, overwriting the stock un-normalized ones.
        action_dim = get_action_dim(action_space)
        self.q_networks = []
        for idx in range(n_critics):
            q_net = nn.Sequential(
                *create_mlp(
                    features_dim + action_dim, 1, net_arch, activation_fn,
                    post_linear_modules=_NORM,
                )
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    """Stock SAC policy that builds LayerNorm-regularized actor + critic networks (D-0027)."""

    def make_actor(self, features_extractor=None) -> LayerNormActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return LayerNormActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor=None) -> LayerNormContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormContinuousCritic(**critic_kwargs).to(self.device)
