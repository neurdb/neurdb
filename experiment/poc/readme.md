# Proof of Concept Experiment

### Experiment 1
We fix the batch size at 100 and vary the total number of samples to 10,000, 100,000, 500,000, and 1,000,000. The execution times for native Python environment inference and in-database inference are shown below.

Batch size: 100 	Device: Intel(R) Xeon(R) W-2133 CPU @ 3.60GHz

**Native Python**

| Number of samples    | 10,000  | 100,000 | 500,000  | 1,000,000 |
| -------------------- | ------- | ------- | -------- | --------- |
| Data loading (ms)    | 27.856  | 173.314 | 897.916  | 1741.987  |
| Model Inference (ms) | 93.245  | 134.439 | 596.596  | 1044.572  |
| Other (ms)           | 0.024   | 0.021   | 0.023    | 0.024     |
| Total (ms)           | 121.125 | 307.774 | 1494.535 | 2786.580  |

**In Database**

| Number of samples    | 10,000  | 100,000 | 500,000  | 1,000,000 |
| -------------------- | ------- | ------- | -------- | --------- |
| Data loading (ms)    | 4.370   | 24.040  | 124.060  | 213.520   |
| Model Inference (ms) | 120.250 | 335.000 | 677.000  | 865.080   |
| Other (ms)           | 346.767 | 380.964 | 397.235  | 358.463   |
| Total (ms)           | 471.387 | 740.004 | 1198.295 | 1437.063  |

![forward_inferece execution time](./image/poc_forward_inference_number_of_samples.png)

<div style="text-align: center; color: grey; font-size: 0.9em;">Figure 1: Forward Inference Execution Time in Native Python and In Database</div>

### Experiment 2

We fix the number of samples to 1,000,000 and vary the batch size to 100, 500, 1,000, 2,000, and 5,000. The execution times for native Python environment inference and in-database inference are shown below.

Number of samples: 1,000,000	Device: Intel(R) Xeon(R) W-2133 CPU @ 3.60GHz

**Native Python**

| Batch size           | 100      | 500      | 1,000    | 5,000    |
| -------------------- | -------- | -------- | -------- | -------- |
| Data loading (ms)    | 1561.645 | 1361.120 | 1373.732 | 1243.088 |
| Model Inference (ms) | 1005.706 | 553.924  | 432.321  | 403.513  |
| Other (ms)           | 0.019    | 0.020    | 0.022    | 0.020    |
| Total (ms)           | 2567.370 | 1915.064 | 1806.075 | 1646.621 |

**In Database**

| Batch size           | 100      | 500      | 1,000    | 5,000    |
| -------------------- | -------- | -------- | -------- | -------- |
| Data loading (ms)    | 226.830  | 228.540  | 314.020  | 257.030  |
| Model Inference (ms) | 1006.840 | 444.000  | 317.110  | 369.300  |
| Other (ms)           | 354.183  | 639.775  | 475.551  | 435.974  |
| Total (ms)           | 1587.853 | 1312.315 | 1106.681 | 1062.304 |

![time vs batch size](./image/poc_time_vs_batch_size.png)
<div style="text-align: center; color: grey; font-size: 0.9em;">Figure 2: Time vs Batch Size</div>

