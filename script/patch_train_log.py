#!/usr/bin/env python3
"""Add training logs: stream_libsvm_dataset.py and armnet builder.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def patch_stream_libsvm():
    p = ROOT / "aiengine/runtime/neurdbrt/dataloader/stream_libsvm_dataset.py"
    text = p.read_text()
    old = (
        "    async def __anext__(self):\n"
        '        """\n'
        "        Wait until the next data is available.\n"
        "        :return: current batch data\n"
        '        """\n'
        '        logger.debug(f"[StreamingDataSet]: reading one data from queue...")\n'
        "        begin_time = time.time()\n"
        "        batch_data = await self.data_cache.get()\n"
        "        end_time = time.time()\n"
        "        self.total_time_fetching += end_time - begin_time\n"
        "        if batch_data is None:\n"
        "            # raise to http response\n"
        '            raise "No data to read after waiting for 10 mins"\n'
        "\n"
        "        # increase the current stage count\n"
        "        self.current_stage_batch_count += 1\n"
        "        if self.current_stage_batch_count >= self.stage_counts[self.current_stage]:"
    )
    new = (
        "    async def __anext__(self):\n"
        '        """\n'
        "        Wait until the next data is available.\n"
        "        :return: current batch data\n"
        '        """\n'
        "        next_batch = self.current_stage_batch_count\n"
        "        total = self.stage_counts.get(self.current_stage, 0) if self.stage_counts else 0\n"
        "        if total > 0:\n"
        "            logger.info(\n"
        '                "waiting for next batch from DB",\n'
        "                stage=str(self.current_stage),\n"
        "                next_batch=next_batch,\n"
        "                total_batches=total,\n"
        "            )\n"
        '        logger.debug(f"[StreamingDataSet]: reading one data from queue...")\n'
        "        begin_time = time.time()\n"
        "        batch_data = await self.data_cache.get()\n"
        "        end_time = time.time()\n"
        "        self.total_time_fetching += end_time - begin_time\n"
        "        if batch_data is None:\n"
        "            # raise to http response\n"
        '            raise "No data to read after waiting for 10 mins"\n'
        "\n"
        "        # increase the current stage count\n"
        "        self.current_stage_batch_count += 1\n"
        "        if total > 0:\n"
        "            logger.info(\n"
        '                "got batch from DB",\n'
        "                stage=str(self.current_stage),\n"
        "                batch_index=self.current_stage_batch_count - 1,\n"
        "                total_batches=total,\n"
        "                wait_sec=round(end_time - begin_time, 2),\n"
        "            )\n"
        "        if self.current_stage_batch_count >= self.stage_counts[self.current_stage]:"
    )
    if old not in text:
        print("stream_libsvm: block not found, skip")
        return False
    p.write_text(text.replace(old, new, 1))
    print("stream_libsvm_dataset.py: OK")
    return True

def patch_armnet_builder():
    p = ROOT / "aiengine/runtime/neurdbrt/model/armnet/builder.py"
    text = p.read_text()
    old = (
        "            train_timestamp = time.time()\n\n"
        "            batch_idx = -1\n"
        "            async for batch in train_loader:\n"
        "                batch_idx += 1\n"
        "                logger.info("
    )
    new = (
        "            train_timestamp = time.time()\n\n"
        "            logger.info(\n"
        '                "train loop started",\n'
        "                epoch=e,\n"
        "                total_epochs=epoch,\n"
        "                train_batch_num=train_batch_num,\n"
        "            )\n"
        "            batch_idx = -1\n"
        "            async for batch in train_loader:\n"
        "                batch_idx += 1\n"
        "                logger.info("
    )
    if old not in text:
        print("armnet builder: block not found, skip")
        return False
    p.write_text(text.replace(old, new, 1))
    print("armnet builder.py: OK")
    return True

if __name__ == "__main__":
    patch_stream_libsvm()
    patch_armnet_builder()
