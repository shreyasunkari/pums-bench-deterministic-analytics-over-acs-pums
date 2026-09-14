# PUMS-BENCH: Deterministic Analytics over ACS PUMS

Solution for the PUMS-BENCH Kaggle competition for ENGR 689 at Texas A&M.

The benchmark contains 38 deterministic analytical questions over the
American Community Survey (ACS) Public Use Microdata Sample (PUMS) for Texas.

## Data

The raw PUMS files are not included in this repository because of their size.

Download the following files from the U.S. Census Bureau:

### 2024 ACS 1-Year PUMS Texas
- `csv_ptx.zip` — person records
- `csv_htx.zip` — household records

https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/

### 2023 ACS 1-Year PUMS Texas
- `csv_ptx.zip` — person records
- `csv_htx.zip` — household records

https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/

Extract the files so the project contains:

```text
data/raw/2024/psam_p48.csv
data/raw/2024/psam_h48.csv
data/raw/2023/psam_p48.csv
data/raw/2023/psam_h48.csv