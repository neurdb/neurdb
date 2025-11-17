# Third-party imports
import torch
import torch.nn as nn

# Local/project imports
from common import BaseConfig

from .logger import plogger


class HybridRankLoss:
    """
    Normalized both classification and ranking losses using exponential moving averages (class_loss_avg, rank_loss_avg)
    with a smoothing factor (0.7) to balance their contributions dynamically.
    """

    def __init__(
        self, cfg: BaseConfig, alpha=0.5, gamma=2.0, class_weights=None, smoothing=0.7
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.smoothing = smoothing
        self.class_loss_avg = 0.02  # Adjusted to match early class_loss
        self.rank_loss_avg = 2.0  # Adjusted to match early rank_loss
        self.cfg = cfg

    def focal_loss(self, logits, targets):
        """
        Measures how well class_logits predict y_multi_label (0s and 1s indicating correct optimizers).
        Focal loss emphasizes hard examples (where predictions are off) via (1 - pt) ** gamma.
        """
        bce_loss = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=self.class_weights
        )(logits, targets)
        pt = torch.exp(-bce_loss)
        return (0.25 * (1 - pt) ** self.gamma * bce_loss).mean()

    def __call__(self, class_logits, reg_times, y_multi_label, y_times):
        class_loss = self.focal_loss(class_logits, y_multi_label)

        masked_times = y_times.masked_fill(~y_multi_label.bool(), float("inf"))
        min_times = masked_times.min(dim=1, keepdim=True)[0]
        # excess_time = torch.relu(reg_times - min_times) * y_multi_label

        # Use absolute deviation instead of ReLU
        excess_time = torch.abs(reg_times - min_times) * y_multi_label
        rank_loss = excess_time.mean() / self.cfg.EXECUTION_TIME_OUT

        self.class_loss_avg = (
            self.smoothing * self.class_loss_avg
            + (1 - self.smoothing) * class_loss.item()
        )
        self.rank_loss_avg = (
            self.smoothing * self.rank_loss_avg
            + (1 - self.smoothing) * rank_loss.item()
        )

        norm_class_loss = class_loss / (self.class_loss_avg + 1e-6)
        norm_rank_loss = rank_loss / (self.rank_loss_avg + 1e-6)

        total_loss = self.alpha * norm_class_loss + (1 - self.alpha) * norm_rank_loss

        plogger.info(
            f"----debug class_loss={class_loss.item()}, rank_loss={rank_loss.item()}, "
            f"norm_class={norm_class_loss.item()}, norm_rank={norm_rank_loss.item()}, "
            f"reg_times_mean={reg_times.mean().item()}, min_times_mean={min_times.mean().item()}"
        )

        # print(f"----debug class_loss={class_loss.item()}, rank_loss={rank_loss.item()}, "
        #       f"norm_class={norm_class_loss.item()}, norm_rank={norm_rank_loss.item()}, "
        #       f"reg_times_mean={reg_times.mean().item()}, min_times_mean={min_times.mean().item()}")
        return total_loss


class HybridLossReg:

    def __init__(
        self, cfg: BaseConfig, alpha=0.5, gamma=2.0, class_weights=None, smoothing=0.7
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.smoothing = smoothing
        self.class_loss_avg = 0.02  # tuned with 4 batches
        self.reg_loss_avg = 0.12  # tuned with 4 batches
        self.cfg = cfg

    def focal_loss(self, logits, targets):
        bce_loss = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=self.class_weights
        )(logits, targets)
        pt = torch.exp(-bce_loss)
        return (0.25 * (1 - pt) ** self.gamma * bce_loss).mean()

    def __call__(self, class_logits, reg_times, y_multi_label, y_times):
        # Check for values exceeding EXECUTION_TIME_OUT
        # if (y_times > EXECUTION_TIME_OUT).any() or (reg_times > EXECUTION_TIME_OUT).any():
        # print(f"Execution times exceed EXECUTION_TIME_OUT={EXECUTION_TIME_OUT}! "
        #                  f"y_times max={y_times.max().item()}, reg_times max={reg_times.max().item()}"))

        # Classification loss
        class_loss = self.focal_loss(class_logits, y_multi_label)

        # Normalize times by EXECUTION_TIME_OUT
        y_times_norm = y_times / self.cfg.EXECUTION_TIME_OUT
        reg_times_norm = (
            reg_times.clamp(min=0, max=self.cfg.EXECUTION_TIME_OUT)
            / self.cfg.EXECUTION_TIME_OUT
        )  # Cap at timeout

        # Regression loss (MSE on normalized times)
        mse_loss = nn.MSELoss(reduction="mean")(reg_times_norm, y_times_norm)
        reg_loss = mse_loss

        # Update EMAs
        self.class_loss_avg = (
            self.smoothing * self.class_loss_avg
            + (1 - self.smoothing) * class_loss.item()
        )
        self.reg_loss_avg = (
            self.smoothing * self.reg_loss_avg + (1 - self.smoothing) * reg_loss.item()
        )

        # Normalize
        norm_class_loss = class_loss / (self.class_loss_avg + 1e-6)
        norm_reg_loss = reg_loss / (self.reg_loss_avg + 1e-6)

        # Combine
        total_loss = self.alpha * norm_class_loss + (1 - self.alpha) * norm_reg_loss

        # print(f"----debug class_loss={class_loss.item()}, reg_loss={reg_loss.item()}, "
        #       f"norm_class={norm_class_loss.item()}, norm_reg={norm_reg_loss.item()}, "
        #       f"reg_times_mean={reg_times.mean().item()}, y_times_mean={y_times.mean().item()}")

        return total_loss
