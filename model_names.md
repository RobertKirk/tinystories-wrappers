# Model Names

This details the learning rates and datasets used for each model in the download: https://drive.google.com/file/d/1yG-T-OjMjtIzlxk-AvDzZHoIsWYnAbKN/view?usp=sharing.

## Deletion models

| Learning Rate | Model name | Dataset |
| ---------- | ------------- | ------- |
|0.00001 |base91M-delete-2023_11_15_00_04_35_349 |filter-mix-Twist-Foreshadowing|
|0.00001 |base91M-delete-2023_11_14_21_05_30_945 |              filter-adv-Twist|
|0.00001 |base91M-delete-2023_11_14_18_06_34_850 |                  filter-Twist|
|0.00010 |base91M-delete-2023_11_14_23_34_52_041 |filter-mix-Twist-Foreshadowing|
|0.00010 |base91M-delete-2023_11_14_20_35_51_288 |              filter-adv-Twist|
|0.00010 |base91M-delete-2023_11_14_17_36_32_483 |                  filter-Twist|

## Recovery Models

| Deletion Model | Model Name |
| ---------- | ------------- |
|base91M-delete-2023_11_14_17_36_32_483/out|base91M-recover-2023_11_15_16_36_04_652|
|base91M-delete-2023_11_14_18_06_34_850/out|base91M-recover-2023_11_15_16_08_51_347|
|base91M-delete-2023_11_14_21_05_30_945/out|base91M-recover-2023_11_15_15_41_52_079|
|base91M-delete-2023_11_14_20_35_51_288/out|base91M-recover-2023_11_15_14_47_42_461|
|base91M-delete-2023_11_14_23_34_52_041/out|base91M-recover-2023_11_15_13_53_17_238|
|base91M-delete-2023_11_15_00_04_35_349/out|base91M-recover-2023_11_15_13_26_06_764|
