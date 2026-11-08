# Data

The dataset is not stored in this repository (the CSV is ~29 MB and is distributed by UNSW for academic use with citation).

1. Go to <https://research.unsw.edu.au/projects/toniot-datasets>.
2. Download the network **Train_Test_datasets** file `train_test_network.csv`.
3. Place it here as `data/train_test_network.csv`.

Expected shape: **211,043 rows × 44 columns** (50,000 normal, 161,043 attack; MITM = 1,043 rows).

## External dataset for notebook 05 (UNSW-NB15)

Notebook 05 tests generalization on UNSW-NB15. Download the partitioned training and testing CSVs
(`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`) from <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
(a mirror is at <https://github.com/InitRoot/UNSW_NB15>, file `UNSW_NB15.zip`) and place them in `data/external/unsw_nb15/`.
That release stores protocol, service and state as numeric codes; the decoding table is in `data/external/unsw_nb15/feature_coding.txt`.

