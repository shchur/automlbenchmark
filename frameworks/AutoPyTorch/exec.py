import copy
import logging
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter("ignore")

from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer, load_timeseries_dataset

log = logging.getLogger(__name__)


metrics_mapping = {
    "mase": "mean_MASE_forecasting",
    "mape": "mean_MAPE_forecasting",
    "mae": "mean_MAE_forecasting",
    "rmse": "mean_MSE_forecasting",
    "mse": "mean_MSE_forecasting",
}


def run(dataset, config):
    train_df, test_df = load_timeseries_dataset(dataset)

    all_series = [ts for _, ts in train_df.groupby(dataset.id_column)]
    y_train = [ts[dataset.target] for ts in all_series]
    start_times = [ts[dataset.timestamp_column].iloc[0] for ts in all_series]

    api = TimeSeriesForecastingTask(
        seed=config.seed,
        ensemble_size=20,
        resampling_strategy=HoldoutValTypes.time_series_hold_out_validation,
        resampling_strategy_args=None,
    )
    api.set_pipeline_options(early_stopping=20, torch_num_threads=config.cores)

    with Timer() as training:
        api.search(
            X_train=None,
            y_train=copy.deepcopy(y_train),
            optimize_metric=metrics_mapping.get(config.metric, "mean_MASE_forecasting"),
            n_prediction_steps=dataset.forecast_horizon_in_steps,
            memory_limit=16 * 1024,
            freq=pd.tseries.frequencies.to_offset(dataset.freq).freqstr,
            start_times=start_times,
            normalize_y=False,
            total_walltime_limit=config.max_runtime_seconds,
            min_num_test_instances=1000,
            budget_type="epochs",
            max_budget=50,
            min_budget=5,
        )

    test_sets = api.dataset.generate_test_seqs()
    with Timer() as predict:
        forecasts = api.predict(test_sets)

    predictions_only = np.concatenate(forecasts)

    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in config.quantile_levels:
        # Probabilistic forecast not supported - repeat the point forecast for each quantile
        optional_columns[str(q)] = predictions_only

    truth_only = test_df[dataset.target].values

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions_only,
        truth=truth_only,
        target_is_encoded=False,
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
        optional_columns=pd.DataFrame(optional_columns),
    )


if __name__ == "__main__":
    call_run(run)
