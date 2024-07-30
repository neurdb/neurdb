from logger.logger import logger
import torch
from typing import List, Union
import torch.nn as nn
import torch.optim as optim
from models.armnet.model.model import ARMNetModel
from models.base.builder import BuilderBase
import time
from utils.date import timeSince
from utils.metrics import AverageMeter, roc_auc_compute_fn
from torch.utils.data import DataLoader
from shared_config.config import DEVICE
from dataloader.steam_libsvm_dataset import StreamingDataSet


class ARMNetModelBuilder(BuilderBase):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._logger = logger.bind(model="ARM-Net")

    def _init_model_arch(self):
        print(f"[_init_model_arch]: Moving model to {DEVICE}")
        if self._model is None:
            self._model = ARMNetModel(
                self._nfield if self._nfield else self.args.nfield,
                self._nfeat if self._nfeat else self.args.nfeat,
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

    def train(
            self,
            train_loader: Union[DataLoader, StreamingDataSet],
            val_loader: Union[DataLoader, StreamingDataSet],
            test_loader: Union[DataLoader, StreamingDataSet],
            epochs: int,
            train_batch_num: int,
            eva_batch_num: int,
            test_batch_num: int,
    ):
        logger = self._logger.bind(task="train")

        # _nfeat, _nfield = self.model_dimension
        # create model
        self._init_model_arch()

        logger.debug("model created with args", **vars(self.args))

        # optimizer
        opt_metric = nn.BCEWithLogitsLoss(reduction="mean").to(DEVICE)
        optimizer = optim.Adam(self._model.parameters(), lr=self.args.lr)

        logger.debug("built the optimziers")

        # gradient clipping
        for p in self.model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
        torch.backends.cudnn.benchmark = True

        logger.debug("register_hook and build cudnn bencnmark")

        patience_cnt = 0
        best_valid_auc = 0.0
        best_test_auc = 0.0
        start_time = time.time()

        logger.debug("start training...")

        for epoch in range(self.args.epoch):
            logger.debug("Epoch start", curr_epoch=epoch, end_at_epoch=self.args.epoch)

            # Training phase
            self._model.train()
            train_time_avg, train_loss_avg, train_auc_avg = (
                AverageMeter(),
                AverageMeter(),
                AverageMeter(),
            )
            train_timestamp = time.time()

            for batch_idx, batch in enumerate(train_loader):
                # logger.debug(
                #     "get batch",
                #     id=batch_idx,
                #     id_shape=batch["id"].shape,
                #     value_shape=batch["value"].shape,
                # )

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
                    logger.debug(
                        "%s",
                        f"Epoch [{epoch:3d}/{self.args.epoch}][{batch_idx:3d}/{len(train_loader)}]\t"
                        f"{train_time_avg.val:.3f} ({train_time_avg.avg:.3f}) AUC {train_auc_avg.val:4f} "
                        f"({train_auc_avg.avg:4f}) Loss {train_loss_avg.val:8.4f} ({train_loss_avg.avg:8.4f})",
                        # epoch={
                        #     "now": epoch,
                        #     "end": self.args.epoch,
                        # },
                        # batch={
                        #     "now": batch_idx,
                        #     "end": len(train_loader),
                        # },
                        # time={"batch": train_time_avg.val, "avg": train_time_avg.avg},
                        # auc={
                        #     "batch": train_auc_avg.val,
                        #     "avg": train_auc_avg.avg,
                        # },
                        # loss={"batch": train_loss_avg.val, "avg": train_loss_avg.avg},
                    )
                if batch_idx + 1 == train_batch_num:
                    break
            logger.debug(
                "Epoch end",
                time=timeSince(s=train_time_avg.sum),
                auc=train_auc_avg.avg,
                loss=train_loss_avg.avg,
            )
            # logger.info(
            #     f"train\tTime {timeSince(s=train_time_avg.sum):>12s} "
            #     f"AUC {train_auc_avg.avg:8.4f} Loss {train_loss_avg.avg:8.4f}"
            # )

            # Validation phase
            valid_auc = self._evaluate(val_loader, opt_metric, "val", eva_batch_num)
            test_auc = self._evaluate(test_loader, opt_metric, "test", test_batch_num)

            # record best auc and save checkpoint
            if valid_auc >= best_valid_auc:
                patience_cnt = 0
                best_valid_auc, best_test_auc = valid_auc, test_auc

                logger.debug(
                    "New best valid auc",
                    epoch=epoch,
                    valid_auc=valid_auc,
                    test_auc=test_auc,
                )
                # logger.info(
                #     f"best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}"
                # )
            else:
                patience_cnt += 1

                logger.debug(
                    "Evaluation early stopped",
                    epoch=epoch - 1,
                    patience_cnt=patience_cnt,
                    valid_auc=valid_auc,
                    test_auc=test_auc,
                )
                # logger.info(f"valid {valid_auc:.4f}, test {test_auc:.4f}")
                # logger.info(
                #     f"Early stopped, {patience_cnt}-th best auc at epoch {epoch - 1}"
                # )

            if patience_cnt >= self.args.patience:
                self._logger.debug(
                    "Evaluation end",
                    epoch=epoch,
                    valid_auc=best_valid_auc,
                    test_auc=best_test_auc,
                )

                # logger.info(
                #     f"Final best valid auc {best_valid_auc:.4f}, with test auc {best_test_auc:.4f}"
                # )
                # break

        self._model.eval()

        self._logger.debug("Train end", time=timeSince(since=start_time))

        self._logger.debug(f"streaming dataloader time usage = {train_loader.total_time_fetching}")
        # logger.info(
        #     f"Total running time for training/validation/test: {timeSince(since=start_time)}"
        # )

    def _evaluate(
            self,
            data_loader: Union[DataLoader, StreamingDataSet],
            opt_metric,
            namespace: str,
            batch_num: int,
    ):
        logger = self._logger.bind(task=namespace)

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
                    logger.debug(
                        f"Epoch [{batch_idx:3d}/{len(data_loader)}]\t"
                        f"{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) "
                        f"Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})"
                    )

                if batch_idx + 1 == batch_num:
                    break
        logger.debug(
            f"Time {timeSince(s=time_avg.sum):>12s} "
            f"AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}"
        )

        logger.debug(f"Evaluate end", time=timeSince(s=time_avg.sum))
        return auc_avg.avg

    def inference(
            self, data_loader: Union[DataLoader, StreamingDataSet], inf_batch_num: int
    ):
        logger = self._logger.bind(task="inference")
        print(f"begin inference for {inf_batch_num} batches ")
        # if this is to load model from the dict,
        if self.args.state_dict_path:
            print("loading model from state dict")
            self._init_model_arch()
            self._model.load_state_dict(torch.load(self.args.state_dict_path))
            logger.info("model loaded", state_dict_path=self.args.state_dict_path)
        else:
            print("loading model from database")

        start_time = time.time()
        predictions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx == inf_batch_num:
                    break
                if torch.cuda.is_available():
                    batch["id"] = batch["id"].cuda(non_blocking=True)
                    batch["value"] = batch["value"].cuda(non_blocking=True)

                y = self._model(batch)
                predictions.append(y.cpu().numpy().tolist())
        print(f"done inference for {inf_batch_num} batches ")
        logger.debug(f"Inference end", time=timeSince(since=start_time))
        return predictions
