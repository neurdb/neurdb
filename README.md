![logo](./doc/logo.jpg)

[![NeurDB Website](https://img.shields.io/badge/Website-neurdb.com-blue)](https://neurdb.com)
[![Github](https://img.shields.io/badge/Github-100000.svg?logo=github&logoColor=white)](https://github.com/neurdb/neurdb)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/neurdb/neurdb)
![GitHub contributors](https://img.shields.io/github/contributors-anon/neurdb/neurdb)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![paper-24-1](https://img.shields.io/badge/DOI-10.1007/s11432--024--4125--9-B31B1B.svg)](https://www.sciengine.com/SCIS/doi/10.1007/s11432-024-4125-9)
[![paper-24-2](https://img.shields.io/badge/arXiv-2408.03013-b31b1b.svg?labelColor=f9f107)](https://arxiv.org/abs/2408.03013)


NeurDB is an AI-powered autonomous data system.

## Installation

Our database is based on the PostgreSQL 16.3 with [doc](https://www.postgresql.org/docs/16/)

### Clone the latest code

```bash
git clone https://github.com/neurdb/neurdb.git
cd neurdb
# Give Docker container write permission
chmod -R 777 .
```

### Build Dockerfile

```bash
bash build.sh --gpu
bash build.sh --cpu
```

Wait until the following prompt shows:

```
Please use 'control + c' to exit the logging print
...
Press CTRL+C to quit
```

### Development

[DB engine dev](./doc/db_dev.md)

[AI engine dev](./doc/ai_dev.md)

## Citation

Our vision paper can be found in:

```
@article{neurdb-scis-24,
  author = {Beng Chin Ooi and
            Shaofeng Cai and
            Gang Chen and
            Yanyan Shen and
            Kian-Lee Tan and
            Yuncheng Wu and
            Xiaokui Xiao and
            Naili Xing and
            Cong Yue and
            Lingze Zeng and
            Meihui Zhang and
            Zhanhao Zhao},
  title  =  {NeurDB: An AI-powered Autonomous Data System},
  journal=  {SCIENCE CHINA Information Sciences},
  year   =  {2024},
  pages  =  {-},
  url    =  {https://www.sciengine.com/SCIS/doi/10.1007/s11432-024-4125-9},
  doi    =  {https://doi.org/10.1007/s11432-024-4125-9}
}
```
