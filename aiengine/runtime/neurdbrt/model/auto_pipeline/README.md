# CtxPipe

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=ffffff)](https://www.python.org/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> CtxPipe: Context-aware Data Preparation Pipeline Construction for Machine Learning (SIGMOD 2025)

![System Architecture](./assets/architecture.svg)

## Overview

CtxPipe is a novel and efficient framework for automated data preparation pipeline construction. It makes use of embedding models to automatically extract context into vectors, which are then integrated into the pipeline construction algorithm via a *gated context plug-in* module.

## Features

- Fully automated data preparation pipeline construction
- Context-aware mechanism: Gated context plug-in module
- Agent optimization strategy for context plug-in: Open-Closed-Gated (OCG) experience replay

## Installation

1. Install [Anaconda](https://www.anaconda.com/download).
1. Clone/download this repo to folder `<ctxpipe_folder>`.
1. Create an Anaconda environment with Python 3.8, and install required packages.

    ```sh
    cd <ctxpipe_folder>
    conda create -n ctxpipe-pt112 python=3.8
    conda activate ctxpipe-pt112

    # create from pip freeze
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

    # or, create from the Anaconda environment file
    conda env create -f conda-env.yml
    ```

1. Clone [GTE-large](https://huggingface.co/thenlper/gte-large) to `<ctxpipe_folder>/../embed/gte-large`.

    ```sh
    git clone https://huggingface.co/thenlper/gte-large ../embed/gte-large
    ```

    You may specify other locations by modifying `CTX_MODEL_PATH` in `<ctxpipe_folder>/ctxpipe/ctx.py`.

1. Open the commented configuration file `<ctxpipe_folder>/config.py` and modify it to reflect the setup of your target environment.
1. Start training/testing.

    ```sh
    # train
    python train.py [<optional_resume_from_step>]
    ```

    ```sh
    # test
    python test.py <start_step> <end_step>

    # or, for scalability test, modify the last line of test.py and then
    python test.py
    ```

## Baseline Setup

| System   | Repository                                                               | Recommended Environment                        |
|----------|--------------------------------------------------------------------------|------------------------------------------------|
| DeepLine | <https://github.com/yuvalhef/gym-deepline>                               | Python 3.6.13, nvidia-tensorflow 1.15.2+nv20.6 |
| HAIPipe  | <https://github.com/ruc-datalab/Haipipe>                                 | Python 3.8.18, torch 1.10.1+cu111              |
| DiffPrep | <https://github.com/chu-data-lab/DiffPrep>                               | Python 3.9.17, torch 1.8.1                     |
| SAGA     | <https://github.com/damslab/reproducibility/tree/master/sigmod2024-SAGA> | OpenJDK 11.0.19                                |

## Dataset

CtxPipe uses the same datasets from [HAIPipe](https://github.com/ruc-datalab/Haipipe) for training, and the datasets from [DiffPrep](https://github.com/chu-data-lab/DiffPrep) and [Deepline](https://github.com/yuvalhef/gym-deepline) for testing.

Please download the datasets at

- <https://github.com/ruc-datalab/Haipipe?tab=readme-ov-file#dataset>
- <https://github.com/chu-data-lab/DiffPrep/tree/main/data>
- <https://github.com/yuvalhef/gym-deepline/tree/master/gym_deepline/envs/datasets/classification/train>

and put them in the corresponding subfolders of `<ctxpipe_folder>/data`.

For example, HAIPipe datasets are put in `<ctxpipe_folder>/data/dataset`, while DiffPrep datasets are put in `<ctxpipe_folder>/data/diffprep_dataset`, etc.

## Development

### Repository Structure

#### Folders

- `ctxpipe`: Main code of the modules.
- `scripts`: Some utility scripts such as experimental data cleaning.

#### Files

- `env.py`: The global initialization and experimental setup.
- `config.py`: The configuration of the training/testing process.
- `deterministic.py`: All settings related to determinism.
- `comp.py`: Involved components, models, and algorithms.
- `util.py`: Some utility functions for filesystem and memory.

### Code Formatting and Linting

All tools are managed by [pre-commit](https://pre-commit.com/).

Run the following command to format and lint code all at once:

```sh
pre-commit run --all-files
```

## License

```
Copyright 2024 CtxPipe Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
