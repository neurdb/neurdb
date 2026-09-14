#!/usr/bin/env python3
"""Four-head hierarchical policy network and online PPO update."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimization.action_vocabulary import canonical_phase
from torch.distributions import Categorical

from ..encoders.state import BaseNetworkTransfer, StructuredState
from .action_space import (
    N_ADAPT,
    N_DEC,
    N_ENUM,
    N_SCHED,
    Transition,
    compute_gae,
)


class BaseNetwork(nn.Module):
    """Shared query/plan encoders and stage-private representation trunks."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.encoder = BaseNetworkTransfer(hidden)
        self.hidden_out = getattr(self.encoder, "hidden_out", hidden)
        self.sched_adapter = nn.Sequential(
            nn.Linear(self.hidden_out, self.hidden_out),
            nn.LayerNorm(self.hidden_out),
            nn.ReLU(),
        )

    def encode_batch(
        self,
        transitions: list[Transition],
        indices: np.ndarray,
        device: torch.device,
        *,
        detach_shared: bool = False,
    ) -> torch.Tensor:
        states = [transitions[index].state for index in indices]
        return self.encoder.encode_structured_batch(
            states,
            device,
            detach_shared=detach_shared,
        )

    def encode_state_obj(
        self,
        state: StructuredState,
        device: torch.device,
    ) -> torch.Tensor:
        return self.encoder.encode_structured_state(state, device)

    def sched_features(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.sched_adapter(hidden)


class HACNetwork(BaseNetwork):
    """Hierarchical actor--critic over Dec, Sched, Enum, and Adapt."""

    def __init__(self, hidden: int = 128):
        super().__init__(hidden)
        self.dec_actor = nn.Linear(self.hidden_out, N_DEC)
        self.dec_critic = nn.Linear(self.hidden_out, 1)
        self.sched_actor = nn.Linear(self.hidden_out, N_SCHED)
        self.sched_critic = nn.Linear(self.hidden_out, 1)
        self.enum_actor = nn.Linear(self.hidden_out, N_ENUM)
        self.enum_critic = nn.Linear(self.hidden_out, 1)
        self.adapt_actor = nn.Linear(self.hidden_out, N_ADAPT)
        self.adapt_cost_head = nn.Linear(self.hidden_out, N_ADAPT)
        self.adapt_critic = nn.Linear(self.hidden_out, 1)


def _mixed_masked_categorical(
    logits: torch.Tensor,
    masks: torch.Tensor,
    temperatures: torch.Tensor,
    exploration_epsilons: torch.Tensor,
    coverage_mixes: Optional[torch.Tensor] = None,
    coverage_probs: Optional[torch.Tensor] = None,
) -> Categorical:
    """Reconstruct the stochastic behavior policy used during collection."""
    if logits.ndim != 2 or masks.shape != logits.shape:
        raise ValueError("logits and masks must have matching [batch, actions] shapes")
    if temperatures.ndim == 1:
        temperatures = temperatures.unsqueeze(-1)
    if exploration_epsilons.ndim == 1:
        exploration_epsilons = exploration_epsilons.unsqueeze(-1)
    if torch.any(temperatures <= 0):
        raise ValueError("behavior-policy temperatures must be greater than zero")
    if torch.any((exploration_epsilons < 0) | (exploration_epsilons >= 1)):
        raise ValueError("behavior-policy exploration epsilon must be in [0, 1)")
    if torch.any(masks.sum(dim=-1) <= 0):
        raise ValueError("each behavior-policy mask must allow at least one action")

    masked_logits = logits / temperatures + (masks - 1) * 1e9
    base_probs = torch.softmax(masked_logits, dim=-1)
    valid_probs = masks / masks.sum(dim=-1, keepdim=True)
    mixed_probs = (
        1.0 - exploration_epsilons
    ) * base_probs + exploration_epsilons * valid_probs
    if coverage_mixes is not None:
        if coverage_mixes.ndim == 1:
            coverage_mixes = coverage_mixes.unsqueeze(-1)
        if torch.any((coverage_mixes < 0) | (coverage_mixes >= 1)):
            raise ValueError("coverage mixture weights must be in [0, 1)")
        if coverage_probs is None or coverage_probs.shape != logits.shape:
            raise ValueError("coverage probabilities must match policy logits")
        coverage_probs = coverage_probs * masks
        totals = coverage_probs.sum(dim=-1, keepdim=True)
        if torch.any(totals <= 0):
            raise ValueError("coverage probabilities need valid positive mass")
        coverage_probs = coverage_probs / totals
        mixed_probs = (
            1.0 - coverage_mixes
        ) * mixed_probs + coverage_mixes * coverage_probs
    return Categorical(probs=mixed_probs)


def _policy_outputs(
    model: HACNetwork,
    hidden: torch.Tensor,
    level: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if level == "dec":
        return model.dec_actor(hidden), model.dec_critic(hidden).squeeze(-1)
    if level == "sched":
        sched_hidden = model.sched_features(hidden)
        return (
            model.sched_actor(sched_hidden),
            model.sched_critic(sched_hidden).squeeze(-1),
        )
    if level == "enum":
        return model.enum_actor(hidden), model.enum_critic(hidden).squeeze(-1)
    if level == "adapt":
        return model.adapt_actor(hidden), model.adapt_critic(hidden).squeeze(-1)
    raise ValueError(f"unknown policy level: {level}")


def ppo_update(
    model: HACNetwork,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    level: str,
    device: torch.device,
    method: str = "standardmdp_rl",
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    ent_coef: float = 0.02,
    vf_coef: float = 0.25,
    gamma: float = 0.99,
    lam: float = 0.95,
    freeze_encoder: bool = False,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> float:
    """Apply one on-policy PPO update to a stage-specific policy head."""
    level = canonical_phase(level)
    if method not in {"hac", "smdp", "standardmdp", "standardmdp_rl"}:
        raise ValueError(f"unsupported policy method: {method}")
    count = len(transitions)
    if count == 0:
        if diagnostics is not None:
            diagnostics.update({"transitions": 0, "updates": 0, "total_loss": 0.0})
        return 0.0

    rewards = np.asarray([transition.reward for transition in transitions])
    dones = np.asarray([transition.done for transition in transitions])
    values = np.asarray([transition.value for transition in transitions])
    durations = np.asarray(
        [transition.duration for transition in transitions],
        dtype=np.float32,
    )
    use_duration = method in {"standardmdp", "standardmdp_rl"} and level == "dec"
    advantages, returns = compute_gae(
        rewards,
        dones,
        values,
        gamma,
        lam,
        durations=durations if use_duration else None,
    )
    returns = np.clip(returns, -10.0, 10.0)
    raw_advantage_mean = float(advantages.mean())
    raw_advantage_std = float(advantages.std())
    return_variance = float(np.var(returns))
    explained_variance = (
        1.0 - float(np.var(returns - values)) / return_variance
        if return_variance > 1e-12
        else 0.0
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actions = (
        torch.from_numpy(np.asarray([transition.action for transition in transitions]))
        .long()
        .to(device)
    )
    old_log_probs = (
        torch.from_numpy(
            np.asarray([transition.log_prob for transition in transitions])
        )
        .float()
        .to(device)
    )
    advantages_tensor = torch.from_numpy(advantages).float().to(device)
    returns_tensor = torch.from_numpy(returns).float().to(device)

    model.train()
    total_loss = 0.0
    updates = 0
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
    }
    timing = os.environ.get("HRL_TIMING") == "1"
    encode_time = forward_time = backward_time = optimizer_time = 0.0

    for _ in range(n_epochs):
        permutation = np.random.permutation(count)
        for start in range(0, count, batch_size):
            indices = permutation[start : start + batch_size]
            train_private_trunk = freeze_encoder and level in {"enum", "adapt"}
            started = time.time()
            hidden = model.encode_batch(
                transitions,
                indices,
                device,
                detach_shared=train_private_trunk,
            )
            if freeze_encoder and not train_private_trunk:
                hidden = hidden.detach()
            if timing:
                encode_time += time.time() - started

            logits, predicted_values = _policy_outputs(model, hidden, level)
            if torch.isnan(logits).any():
                continue
            masks = torch.stack(
                [
                    torch.from_numpy(transitions[index].mask).float().to(device)
                    for index in indices
                ]
            )
            temperatures = torch.tensor(
                [transitions[index].temperature for index in indices],
                dtype=torch.float32,
                device=device,
            )
            exploration = torch.tensor(
                [transitions[index].exploration_epsilon for index in indices],
                dtype=torch.float32,
                device=device,
            )
            coverage_mixes = torch.tensor(
                [transitions[index].coverage_mix for index in indices],
                dtype=torch.float32,
                device=device,
            )
            coverage_probs = torch.stack(
                [
                    torch.from_numpy(
                        transitions[index].coverage_probs
                        if transitions[index].coverage_probs is not None
                        else transitions[index].mask
                        / max(float(transitions[index].mask.sum()), 1.0)
                    )
                    .float()
                    .to(device)
                    for index in indices
                ]
            )

            if timing:
                started = time.time()
            distribution = _mixed_masked_categorical(
                logits,
                masks,
                temperatures,
                exploration,
                coverage_mixes,
                coverage_probs,
            )
            batch_actions = actions[indices]
            new_log_probs = distribution.log_prob(batch_actions)
            entropy = distribution.entropy()
            log_ratio = new_log_probs - old_log_probs[indices]
            unclamped_ratio = torch.exp(log_ratio)
            approx_kl = ((unclamped_ratio - 1.0) - log_ratio).mean()
            clip_fraction = (torch.abs(unclamped_ratio - 1.0) > clip_eps).float().mean()
            ratio = torch.clamp(unclamped_ratio, 0.01, 100.0)
            batch_advantages = advantages_tensor[indices]
            surrogate = ratio * batch_advantages
            clipped_surrogate = (
                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            )
            policy_loss = -torch.min(surrogate, clipped_surrogate).mean()
            value_loss = F.mse_loss(
                predicted_values,
                returns_tensor[indices].clamp(-10.0, 10.0),
            )
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()
            if torch.isnan(loss) or loss.item() > 1e6:
                continue
            if timing:
                forward_time += time.time() - started
                started = time.time()

            optimizer.zero_grad()
            loss.backward()
            if timing:
                backward_time += time.time() - started
                started = time.time()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if timing:
                optimizer_time += time.time() - started

            total_loss += loss.item()
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy.mean().item())
            totals["approx_kl"] += float(approx_kl.item())
            totals["clip_fraction"] += float(clip_fraction.item())
            totals["gradient_norm"] += float(gradient_norm.item())
            updates += 1

    if timing:
        print(
            f"  [timing/{level}] encode={encode_time:.2f}s "
            f"fwd={forward_time:.2f}s bwd={backward_time:.2f}s "
            f"optim={optimizer_time:.2f}s batches={updates}",
            flush=True,
        )
    average_loss = total_loss / max(updates, 1)
    if diagnostics is not None:
        diagnostics.update(
            {
                "transitions": count,
                "updates": updates,
                "reward_sum": float(rewards.sum()),
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "raw_advantage_mean": raw_advantage_mean,
                "raw_advantage_std": raw_advantage_std,
                "explained_variance": explained_variance,
                "total_loss": average_loss,
                **{key: value / max(updates, 1) for key, value in totals.items()},
            }
        )
    return average_loss
