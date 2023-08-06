## Reproducing the experiments
This folder contains the instructions for reproducing the result from the paper **AutoGluon–TimeSeries: AutoML for Probabilistic Time Series Forecasting** published at AutoML Conference 2023.

### 1. Install AutoMLBenchmark
1. Create a Python virtual environment for the package where the `automlbenchmark` package will be installed. This can be done with `venv`
    ```bash
    cd ..  # change to the root folder of the `automlbenchark` repository
    python -m venv amlb_venv
    source amlb_venv/bin/activate
    ```
    or `conda`
    ```bash
    cd ..  # change to the root folder of the `automlbenchark` repository
    conda create -n  amlb_venv python=3.9
    conda activate amlb_venv
    ```
2. Install `automlbenchmark` with dependencies
    ```bash
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```
3. Change back to the `autogluon_timeseries_automl23` folder
    ```bash
    cd autogluon_timeseries_automl23
    ```

### 2. Download the datasets
1. Activate the virtual environment created in the previous step and install the necessary dependencies for downloading the datasets.
    ```bash
    python -m pip install gluonts==0.13.2 pandas orjson pyyaml xlrd awscli
    ```
2. Download the raw dataset for the M3 competition
    ```bash
    wget https://forecasters.org/data/m3comp/M3C.xls -P ~/.gluonts/datasets
    ```
3. Download all datasets using GluonTS and save them in CSV format
    ```bash
    python download_datasets.py
    ```
    By default, the datasets are stored in `../datasets/`.

4. Generate the task definition files in YAML format
    ```bash
    python generate_task_configs.py
    ```
    By default, the task definitions are stored in `~/.config/automlbenchmark/benchmarks/`.
5. Copy the `config.yaml` file to the custom configs folder
    ```bash
    cp config.yaml ~/.config/automlbenchmark/
    ```

### 3. Verify that the installation
Run the `SeasonalNaive` method on the `nn5_dataset`
```bash
python ../runbenchmark.py SeasonalNaive quantile_forecast 10m16c -u ~/.config/automlbenchmark/ -t nn5_daily
```
You should see output similar to the following
```
Summing up scores for current run:
       id      task  fold     framework constraint    result  metric  duration  seed
nn5_daily nn5_daily     0 SeasonalNaive      4h16c -0.292439 neg_wql      10.7     0
```

### 4. Run the benchmark
Use the following command to run the experiments locally
```bash
python ../runbenchmark.py [FRAMEWORK] [BENCHMARK] [CONSTRAINT] -u ~/.config/automlbenchmark/
```

Available frameworks
- `AutoGluon_bestquality`
- `AutoPyTorch`
- `AutoARIMA`
- `AutoETS`
- `AutoTheta`
- `SeasonalNaive`
- `StatEnsemble`
- `DeepAR`
- `TFT`

Available benchmarks (=files with task definitions)
- `point_forecast` - 29 datasets, `MASE` evaluation metric
- `quantile_forecast` - 29 datasets, `WQL` evaluation metric on 9 quantile levels `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`.

Available constraints
- `4h16c` - 4 hours, 16 CPU cores, should be used for all experiments

#### Ablation studies
To reproduce the ablation study, where some models are removed, set `[FRAMEWORK]` to one of
- `AutoGluon_NoEnsemble`
- `AutoGluon_NoDeepModels`
- `AutoGluon_NoStatModels`
- `AutoGluon_NoTreeModels`

To reproduce the ablation study, where the training time is reduced, set `[CONSTRAINT]` to one of `1h16c` (1 hour) or `10m16c` (10 minutes).

### (Optional) 4. Run experiments on AWS

**WARNING**: AMLB does not limit expenses! The AWS integration lets your easily conduct massively parallel evaluations. The AutoML Benchmark does not in any way restrict the total costs you can make on AWS.


1. Complete the steps described in https://openml.github.io/automlbenchmark/docs/using/aws/ to configure `automlbenchmark` to run on AWS.
    -  As a part of this process, you will need to create an S3 bucket for storing datasets & results. For example, this could be `s3://automl-benchmark-mybucket`. Remember this name.

2. Upload the datasets to your S3 bucket
    ```bash
    aws s3 cp --recursive ../datasets/ [S3_DATASETS_PREFIX]
    ```
    Here `[S3_DATASETS_PREFIX]` is the name of the folder in your bucket, where the datasets will be stored. This could be `s3://automl-benchmark-mybucket/datasets/`

3. Generate the task configs
    ```bash
    python generate_task_configs.py -d [S3_DATASETS_PREFIX]
    ```

4. Replace line 35 in `./config.yaml` with the name of the S3 bucket where the results will be saved.
    For example, if in the previous step you saved the data to `s3://automl-benchmark-mybucket/datasets/`, you should set
    ```yaml
        bucket: automl-benchmark-mybucket
    ```

5. Copy the updated `config.yaml` file to the custom configs folder
    ```bash
    cp config.yaml ~/.config/automlbenchmark/
    ```

6. Run a sanity check to ensure that configuration is correct.
    ```bash
    python ../runbenchmark.py SeasonalNaive quantile_forecast_aws 10m16c -u ~/.config/automlbenchmark/ -t nn5_daily -m aws
    ```


7. Run all experiments with the following command
    ```bash
    python ../runbenchmark.py [FRAMEWORK] [BENCHMARK] [CONSTRAINT] -u ~/.config/automlbenchmark/ -m aws -p [NUM_PARALLEL_JOBS]
    ```

### 5. Aggregating the results
1. Install `autogluon-benchmark`
```bash
python -m pip install git+https://github.com/Innixma/autogluon-benchmark.git@27bd462a30f0d4ad395fb7f5b43ac45d4111c728
```

2. Combine the results into a single CSV file
    - If you ran the experiments locally, the results will be stored in `../results/`.

    - If you ran the experiments using AWS, the results will be stored in `s3://[YOUR-S3-BUCKET]/ec2/autogluon-timeseries-automl23/`.

        In this case, you can easily combine the results into a single CSV file using the following command
        ```bash
        git clone https://github.com/Innixma/autogluon-benchmark.git
        cd autogluon-benchmark/
        python python scripts/aggregate_openml_results.py --s3_bucket [YOUR-S3-BUCKET] --s3_prefix ec2/ --version_name autogluon-timeseries-automl23
        ```

3. Run the contents of the notebook `aggregate_results.ipynb`.
