# NeurDB Python Server

## Installation

```sh
pip install -r requirement.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## CLI

```text
usage: cli.py [-h] [-c CONFIG] [-l LOGFILE] [-t | --train | --no-train] [-i | --inference | --no-inference]
              [-m MODEL_ID]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
  -l LOGFILE, --logfile LOGFILE
  -t, --train, --no-train
  -i, --inference, --no-inference
  -m MODEL_ID, --model-id MODEL_ID
```

### Examples

Train & Inference

```sh
python cli.py -ti
```

Finetune existing model (if MODEL_ID exists)

```sh
python cli.py -ti -m MODEL_ID
```

Inference only

```sh
python cli.py -i -m MODEL_ID
```
