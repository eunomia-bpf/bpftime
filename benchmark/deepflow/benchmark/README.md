# How to test?

Directly run `run.py`

```console
root@mnfe-pve:~/bpftime-evaluation/deepflow-runtime.casgstatus/benchmark# python3 run.py 
usage: run.py [-h] -t {no-probe,kernel-uprobe,no-uprobe,user-uprobe} [--http] [--https]
              [--wrk-thread WRK_THREAD] [--wrk-conn WRK_CONN]
run.py: error: the following arguments are required: -t/--type
```

Certainly! I will provide the detailed data in two Markdown tables, one for HTTP and one for HTTPS. Each table will have the sub-key types (like 'kernel-uprobe', 'no-uprobe', etc.) as columns, and the sizes (1-256) as rows. Each cell will contain the average request and average transfer values, separated by a slash.

## result

### HTTPS Averages

| Size | kernel-uprobe | no-probe | no-uprobe | user-uprobe |
|------|---------------|----------|-----------|-------------|
| 1    | 76343.52 / 83.15 | 159687.78 / 173.92 | 129640.17 / 141.19 | 88601.28 / 96.50 |
| 2    | 77731.49 / 160.57 | 153616.17 / 317.32 | 128003.31 / 264.41 | 92023.17 / 190.09 |
| 4    | 59957.20 / 242.05 | 113415.38 / 457.86 | 98361.25 / 397.09 | 70625.88 / 285.12 |
| 16   | 50640.45 / 797.90 | 92878.51 / 1463.30 | 82047.01 / 1291.26 | 58080.14 / 915.36 |
| 128  | 26226.10 / 3282.94 | 34996.71 / 4378.62 | 31068.57 / 3887.10 | 27718.18 / 3469.31 |
| 256  | 16103.92 / 4027.39 | 19931.38 / 4984.83 | 17540.86 / 4387.84 | 16755.62 / 4193.28 |

### HTTP Averages

| Size | kernel-uprobe | no-probe | no-uprobe | user-uprobe |
|------|---------------|----------|-----------|-------------|
| 1    | 118499.38 / 129.06 | 256345.72 / 279.19 | 196381.58 / 213.88 | 122848.50 / 133.79 |
| 2    | 116609.60 / 240.88 | 250357.85 / 517.15 | 192184.16 / 396.99 | 121360.12 / 250.69 |
| 4    | 98524.44 / 397.73 | 206790.04 / 834.79 | 162954.94 / 657.83 | 102417.73 / 413.45 |
| 16   | 85274.67 / 1343.49 | 173960.99 / 2742.27 | 133481.71 / 2104.32 | 93158.75 / 1467.39 |
| 128  | 51362.17 / 6426.62 | 83321.26 / 10425.34 | 72278.05 / 9043.97 | 63630.56 / 7961.60 |
| 256  | 30745.06 / 7691.26 | 50518.03 / 12635.14 | 45729.22 / 11439.10 | 42739.36 / 10688.51 |

These tables present the complete data for all sizes (1-256) under each sub-key for both HTTPS and HTTP. The values represent the average request and transfer metrics, which give insights into the network performance characteristics for each size and protocol type.
