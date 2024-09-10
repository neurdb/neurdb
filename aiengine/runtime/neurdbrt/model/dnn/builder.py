import time

import torch
import torch.nn as nn
import torch.optim as optim
from neurdbrt.config import DEVICE
from neurdbrt.dataloader import StreamingDataSet
from neurdbrt.log import logger
from neurdbrt.utils.date import time_since
from neurdbrt.utils.metrics import AverageMeter, roc_auc_compute_fn

from ..base import BuilderBase
from .model import DNNModel


class DNNModelBuilder(BuilderBase):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._logger = logger.bind(model="DNN")

    def _init_model_arch(self):
        print(f"[_init_model_arch]: Moving model to {DEVICE}")
        if self._model is None:
            self._model = DNNModel(
                self._nfield if self._nfield else self.args.nfield,
                self._nfeat if self._nfeat else self.args.nfeat,
                self.args.nemb,
                self.args.mlp_nlayer,
                self.args.mlp_nhid,
                self.args.dropout,
            ).to(DEVICE)

    async def train(
            self,
            train_loader: StreamingDataSet,
            val_loader: StreamingDataSet,
            test_loader: StreamingDataSet,
            epoch: int,
            train_batch_num: int,
            eva_batch_num: int,
            test_batch_num: int,
    ):
        logger = self._logger.bind(task="train")

        # create model
        self._init_model_arch()

        logger.info("model created with args", **vars(self.args))

        # if this is to load model from the dict,
        if self.args.state_dict_path:
            print("loading model from state dict")
            self._init_model_arch()
            self._model.load_state_dict(torch.load(self.args.state_dict_path))
            logger.info("model loaded", state_dict_path=self.args.state_dict_path)
        else:
            print("loading model from database")

        # optimizer
        opt_metric = nn.BCEWithLogitsLoss(reduction="mean").to(DEVICE)
        optimizer = optim.Adam(self._model.parameters(), lr=self.args.lr)

        logger.info("built the optimziers")

        # gradient clipping
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        torch.backends.cudnn.benchmark = True

        logger.info("register_hook and build cudnn bencnmark")

        patience_cnt = 0
        best_valid_auc = 0.0
        best_test_auc = 0.0
        start_time = time.time()

        logger.info("start training...")

        for epoch in range(self.args.epoch):
            logger.info("Epoch start", curr_epoch=epoch, end_at_epoch=self.args.epoch)

            # Training phase
            self._model.train()
            train_time_avg, train_loss_avg, train_auc_avg = (
                AverageMeter(),
                AverageMeter(),
                AverageMeter(),
            )
            train_timestamp = time.time()

            batch_idx = -1
            async for batch in train_loader:
                batch_idx += 1

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
                        "%s",
                        f"Epoch [{epoch:3d}/{self.args.epoch}][{batch_idx:3d}/{len(train_loader)}]\t"
                        f"{train_time_avg.val:.3f} ({train_time_avg.avg:.3f}) AUC {train_auc_avg.val:4f} "
                        f"({train_auc_avg.avg:4f}) Loss {train_loss_avg.val:8.4f} ({train_loss_avg.avg:8.4f})",
                    )
                if batch_idx + 1 == train_batch_num:
                    break

            logger.info(
                "Epoch end",
                time=time_since(s=train_time_avg.sum),
                auc=train_auc_avg.avg,
                loss=train_loss_avg.avg,
            )

            # Validation phase
            valid_auc = await self._evaluate(
                val_loader, opt_metric, "val", eva_batch_num
            )
            test_auc = await self._evaluate(
                test_loader, opt_metric, "test", test_batch_num
            )

            # record best auc and save checkpoint
            if valid_auc >= best_valid_auc:
                patience_cnt = 0
                best_valid_auc, best_test_auc = valid_auc, test_auc

                logger.info(
                    "New best valid auc",
                    epoch=epoch,
                    valid_auc=valid_auc,
                    test_auc=test_auc,
                )

            else:
                patience_cnt += 1

                logger.info(
                    "Evaluation early stopped",
                    epoch=epoch - 1,
                    patience_cnt=patience_cnt,
                    valid_auc=valid_auc,
                    test_auc=test_auc,
                )

            if patience_cnt >= self.args.patience:
                self._logger.info(
                    "Evaluation end",
                    epoch=epoch,
                    valid_auc=best_valid_auc,
                    test_auc=best_test_auc,
                )

        self._model.eval()

        self._logger.info("Train end", time=time_since(since=start_time))

        if isinstance(train_loader, StreamingDataSet):
            self._logger.info(
                f"streaming dataloader time usage = {train_loader.total_time_fetching}"
            )

    async def _evaluate(
            self,
            data_loader: StreamingDataSet,
            opt_metric,
            namespace: str,
            batch_num: int,
    ):
        logger = self._logger.bind(task=namespace)

        self._model.eval()

        time_avg, loss_avg, auc_avg = AverageMeter(), AverageMeter(), AverageMeter()
        timestamp = time.time()

        with torch.no_grad():
            batch_idx = -1
            async for batch in data_loader:
                batch_idx += 1

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
                        f"Epoch [{batch_idx:3d}/{len(data_loader)}]\t"
                        f"{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) "
                        f"Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})"
                    )

                if batch_idx + 1 == batch_num:
                    break
        logger.info(
            f"Time {time_since(s=time_avg.sum):>12s} "
            f"AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}"
        )

        logger.info(f"Evaluate end", time=time_since(s=time_avg.sum))
        return auc_avg.avg

    async def inference(self, data_loader: StreamingDataSet, inf_batch_num: int):
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
            batch_idx = -1
            async for batch in data_loader:
                batch_idx += 1

                if torch.cuda.is_available():
                    batch["id"] = batch["id"].cuda(non_blocking=True)
                    batch["value"] = batch["value"].cuda(non_blocking=True)

                y = self._model(batch)
                predictions.append(y.cpu().numpy().tolist())
                logger.info(f"done batch for {batch_idx}, total {inf_batch_num} ")
                if batch_idx + 1 == inf_batch_num:
                    break

        logger.info("Done inference for {inf_batch_num} batches ")
        logger.info("---- Inference end ---- ", time=time_since(since=start_time))
        return predictions
