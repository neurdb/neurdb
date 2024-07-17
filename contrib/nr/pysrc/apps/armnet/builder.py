from logger.logger import logger
import torch
import torch.nn as nn
import torch.optim as optim
from apps.armnet.model.model import ARMNetModel
from apps.base.builder import BuilderBase
import time
from utils.date import timeSince
from utils.metrics import AverageMeter, roc_auc_compute_fn
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ARMNetModelBuilder(BuilderBase):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._model = None

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader
    ):
        # _nfeat, _nfield = self.model_dimension

        # create model
        self._model = ARMNetModel(
            self._nfield,
            self._nfeat,
            self.args.nemb,
            self.args.nattn_head,
            self.args.alpha,
            self.args.h,
            self.args.mlp_nlayer,
            self.args.mlp_nhid,
            self.args.dropout,
            self.args.ensemble,
            self.args.dnn_nlayer,
            self.args.dnn_nhid,
        ).to(DEVICE)
        logger.info(vars(self.args))

        # optimizer
        opt_metric = nn.BCEWithLogitsLoss(reduction="mean").to(DEVICE)
        optimizer = optim.Adam(self._model.parameters(), lr=self.args.lr)

        # gradient clipping
        for p in self.model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
        torch.backends.cudnn.benchmark = True

        patience_cnt = 0
        best_valid_auc = 0.0
        best_test_auc = 0.0
        start_time = time.time()

        for epoch in range(self.args.epoch):
            logger.info(f"Epoch [{epoch:3d}/{self.args.epoch:3d}]")

            # Training phase
            self._model.train()
            train_time_avg, train_loss_avg, train_auc_avg = (
                AverageMeter(),
                AverageMeter(),
                AverageMeter(),
            )
            train_timestamp = time.time()

            for batch_idx, batch in enumerate(train_loader):
                target = batch["y"]
                if torch.cuda.is_available():
                    batch["id"] = batch["id"].cuda(non_blocking=True)
                    batch["value"] = batch["value"].cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                y = self._model(batch)
                loss = opt_metric(y, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                auc = roc_auc_compute_fn(y, target)
                train_loss_avg.update(loss.item(), target.size(0))
                train_auc_avg.update(auc, target.size(0))

                train_time_avg.update(time.time() - train_timestamp)
                train_timestamp = time.time()

                if batch_idx % self.args.report_freq == 0:
                    logger.info(
                        f"Epoch [{epoch:3d}/{self.args.epoch}][{batch_idx:3d}/{len(train_loader)}]\t"
                        f"{train_time_avg.val:.3f} ({train_time_avg.avg:.3f}) AUC {train_auc_avg.val:4f} "
                        f"({train_auc_avg.avg:4f}) Loss {train_loss_avg.val:8.4f} ({train_loss_avg.avg:8.4f})"
                    )

            logger.info(
                f"train\tTime {timeSince(s=train_time_avg.sum):>12s} "
                f"AUC {train_auc_avg.avg:8.4f} Loss {train_loss_avg.avg:8.4f}"
            )

            # Validation phase
            valid_auc = self._evaluate(val_loader, opt_metric, "val")
            test_auc = self._evaluate(test_loader, opt_metric, "test")

            # record best auc and save checkpoint
            if valid_auc >= best_valid_auc:
                patience_cnt = 0
                best_valid_auc, best_test_auc = valid_auc, test_auc
                logger.info(
                    f"best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}"
                )
            else:
                patience_cnt += 1
                logger.info(f"valid {valid_auc:.4f}, test {test_auc:.4f}")
                logger.info(
                    f"Early stopped, {patience_cnt}-th best auc at epoch {epoch - 1}"
                )
            if patience_cnt >= self.args.patience:
                logger.info(
                    f"Final best valid auc {best_valid_auc:.4f}, with test auc {best_test_auc:.4f}"
                )
                break

        self._model.eval()
        logger.info(
            f"Total running time for training/validation/test: {timeSince(since=start_time)}"
        )
        
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value

    def _evaluate(self, data_loader: DataLoader, opt_metric, namespace="val"):
        self._model.eval()

        time_avg, loss_avg, auc_avg = AverageMeter(), AverageMeter(), AverageMeter()
        timestamp = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                target = batch["y"]
                if torch.cuda.is_available():
                    batch["id"] = batch["id"].cuda(non_blocking=True)
                    batch["value"] = batch["value"].cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                y = self._model(batch)
                loss = opt_metric(y, target)

                auc = roc_auc_compute_fn(y, target)
                loss_avg.update(loss.item(), target.size(0))
                auc_avg.update(auc, target.size(0))

                time_avg.update(time.time() - timestamp)
                timestamp = time.time()

                if batch_idx % self.args.report_freq == 0:
                    logger.info(
                        f"{namespace}\tEpoch [{batch_idx:3d}/{len(data_loader)}]\t"
                        f"{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) "
                        f"Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})"
                    )

        logger.info(
            f"{namespace}\tTime {timeSince(s=time_avg.sum):>12s} "
            f"AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}"
        )
        return auc_avg.avg

    def inference(self, data_loader: DataLoader):
        start_time = time.time()
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                if torch.cuda.is_available():
                    batch["id"] = batch["id"].cuda(non_blocking=True)
                    batch["value"] = batch["value"].cuda(non_blocking=True)

                y = self._model(batch)
                predictions.append(y.cpu().numpy().tolist())

        logger.info(f"Total running time for inference: {timeSince(since=start_time)}")
        return predictions
