# rel-avito dataset + workload

Raw, **multi-table** rel-avito dataset and the SQL workload used by the
end-to-end horizon-sweep experiment (see `../../.local/prompts/exp.md`). The data
is imported into NeurDB and features / labels are built database-natively via
point-in-time (PIT) joins (see `workloads/`).

> The raw dataset (`rel-avito-db/`) and CSV dumps (`workloads/dump/`) are
> git-ignored. Do **not** commit datasets -- re-download them with the steps
> below. Tracked here: the workload SQL/scripts and the committed unit-test
> fixture `w_task_1.csv`.

## Layout

```
test/avito/
├── README.md                  # this file
├── w_task_1.csv               # committed unit-test fixture (1d horizon task table)
├── workloads/                 # SQL + scripts that build w_task_<h> (tracked)
│   ├── tool_cutoffs.sql  01_label_adctr.sql  02_features_adctr_pit.sql  03_task_table.sql
│   ├── tool_feat_cache_init.sql  run_avito.sh  bench.sh  dump_tasks.sh  README.md
│   └── dump/                   # scratch CSV dumps from dump_tasks.sh (ignored)
└── rel-avito-db/               # rel-avito PREBUILT cleaned DB (form B, ignored)
    ├── db.zip
    └── db/                     # 8 cleaned single-file parquet tables
        ├── AdsInfo.parquet  Category.parquet  Location.parquet  UserInfo.parquet
        └── SearchInfo.parquet  SearchStream.parquet  VisitStream.parquet  PhoneRequestsStream.parquet
```

### Versions: raw vs prebuilt (same 100k subsample, two forms)

`rel-avito` is a **subsample** of the original Avito Context Ad Clicks dataset
(subsampled to ~100k users; `UserInfo` ≈ 98k). RelBench does not distribute a
full-size version. It ships the same subsample in two forms:

| Form | File | Size | sha256 | State |
|------|------|-----:|--------|-------|
| **A. raw source** | `rel-avito-db/rel-avito-raw-100k.zip` | 495 MB | `ad4fc178…d7929` | uncleaned; needs `make_db()` processing |
| **B. prebuilt DB** | `download/rel-avito/db.zip` | 347 MB | `274e6922…d77058` | cleaned `Database` (what `get_dataset(download=True)` pulls) |

**We load form B** (prebuilt) into NeurDB for parity with the RelBench `AdCTRTask`
definition. In form B: `Params`/`SearchParams` are dropped, date columns are real
`TIMESTAMP`, and rows are filtered to `>= 2015-04-25` (range 2015-04-25 .. 2015-05-20).

## rel-avito

- **Source:** RelBench (subsampled 100k version of the Avito Context Ad Clicks dataset).
  Original: https://www.kaggle.com/competitions/avito-context-ad-clicks
- **Download URL:** `https://relbench.stanford.edu/data/rel-avito-raw-100k.zip`
- **Size:** ~495 MB (zip); `sha256 = ad4fc1789d8a5073ea449049888c671899525c9a8a42359ca75d1f17d04d7929`
- **Time range:** `SearchDate` spans 2015-04-25 .. 2015-05-20.
  RelBench splits: `val_timestamp = 2015-05-08`, `test_timestamp = 2015-05-14`.

### How to download

Option A — direct download (raw form A):

```bash
mkdir -p test/avito/rel-avito-db && cd test/avito/rel-avito-db
curl -O https://relbench.stanford.edu/data/rel-avito-raw-100k.zip
# verify integrity (must match the sha256 above)
sha256sum rel-avito-raw-100k.zip
unzip -q rel-avito-raw-100k.zip      # -> avito_100k_integ_test/
```

Option B — prebuilt cleaned DB (what we use for loading into NeurDB):

```bash
cd test/avito/rel-avito-db
curl -O https://relbench.stanford.edu/download/rel-avito/db.zip
sha256sum db.zip      # 274e692295027a753063b9201815a9d2dea94d4cda968be81be936f546d77058
unzip -q db.zip       # -> db/<Table>.parquet  (8 cleaned single-file tables)
```

Option C — via the RelBench Python package (downloads form B under the hood):

```bash
pip install relbench pooch pyarrow
python -c "from relbench.datasets import get_dataset; get_dataset('rel-avito', download=True)"
# caches under pooch's os_cache, e.g. ~/.cache/relbench/
```

### Tables (prebuilt cleaned DB, form B)

| Table                 | Rows       | Key columns / notes                                             |
|-----------------------|-----------:|-----------------------------------------------------------------|
| `AdsInfo`             |  5,960,558 | pk `AdID`; fk `LocationID`, `CategoryID`; `Price`, `Title`, `IsContext` |
| `Category`            |         68 | pk `CategoryID`; `Level`, `ParentCategoryID`, `SubcategoryID`    |
| `Location`            |      3,512 | pk `LocationID`; `RegionID`, `CityID`, `Level`                   |
| `UserInfo`            |     98,250 | pk `UserID`; `UserAgentID`, `UserAgentOSID`, `UserDeviceID`, `UserAgentFamilyID` |
| `SearchInfo`          |  2,579,289 | pk `SearchID`; fk `UserID`, `LocationID`, `CategoryID`; time `SearchDate`; `SearchQuery`, `IsUserLoggedOn`, `IPID` |
| `SearchStream`        |  9,254,702 | fk `SearchID`, `AdID`; time `SearchDate`; `Position`, `HistCTR`, `IsClick`, `ObjectType` |
| `VisitStream`         |  6,454,562 | fk `UserID`, `AdID`; time `ViewDate`; `IPID`                     |
| `PhoneRequestsStream` |    302,974 | fk `UserID`, `AdID`; time `PhoneRequestDate`; `IPID`            |

(Raw form A has slightly higher counts and additional dict columns `Params`/`SearchParams`.)

Notes for loading into NeurDB:

- We load **form B** (`rel-avito-db/db/*.parquet`): already cleaned — dates are real
  `TIMESTAMP`, `Params`/`SearchParams` removed, rows filtered to `>= 2015-04-25`.
- If you instead load **form A** (raw): date columns are `VARCHAR` (cast to `timestamp`),
  and you should drop `Params`/`SearchParams` and filter to `>= 2015-04-25` yourself to
  match the RelBench task tables.
- The entity table for the CTR task is `AdsInfo` (`AdID`); the main event table is
  `SearchStream` (impressions + `IsClick`).

## rel-hm (later)

`rel-hm` (H&M) is not auto-downloadable: the raw CSVs come from a Kaggle competition
and require a Kaggle account/API key.

- **Source:** https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
- Tables: `article`, `customer`, `transactions` (time `t_dat`).

```bash
# requires `kaggle` CLI configured with an API token
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
```
