import logging
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter('ignore')

from pycaret.time_series import TSForecastingExperiment
from joblib.externals.loky import get_reusable_executor

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer, load_timeseries_dataset

log = logging.getLogger(__name__)


def run(dataset, config):
    train_df, test_df = load_timeseries_dataset(dataset)

    coverage, point_alpha, column_rename_map = get_probabilistic_forecast_config(config.quantile_levels)

    framework_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    enable_hpo = framework_params.pop("enable_hpo", False)
    max_num_folds = framework_params.pop("max_num_folds", 3)

    results = []
    training_time = 0.0
    predict_time = 0.0
    for item_id, ts in train_df.groupby(dataset.id_column):
        experiment = TSForecastingExperiment()
        ts = ts.drop(dataset.id_column, axis=1).set_index(dataset.timestamp_column)
        # Reduce number of folds for time series that are too short
        num_folds = min(len(ts) // dataset.forecast_horizon_in_steps - 1, max_num_folds)
        experiment.setup(
            data=ts,
            fh=dataset.forecast_horizon_in_steps,
            seasonal_period=dataset.seasonality,
            session_id=123,
            experiment_name=f"series_{item_id}",
            coverage=coverage,
            point_alpha=point_alpha,  # ensure that probabilistic forecast is generated
            verbose=False,
            fold=num_folds,
            **framework_params
        )
        with Timer() as training:
            best = experiment.compare_models(n_select=1, sort=get_eval_metric(config.metric), verbose=False)
            if enable_hpo:
                best = experiment.tune_model(best)
            log.info(f"Best model for item {item_id}: {best}")
            final = experiment.finalize_model(best)

        training_time += training.duration

        with Timer() as predict:
            pred = experiment.predict_model(final, return_pred_int=True)

        predict_time += predict.duration

        pred["item_id"] = item_id
        results.append(pred)

    predictions = pd.concat(results)

    predictions_only = predictions["y_pred"].values
    truth_only = test_df[dataset.target].values

    # Sanity check - make sure predictions are ordered correctly
    if (predictions["item_id"].values != test_df[dataset.id_column].values).any():
        raise AssertionError("item_id column for predictions doesn't match test data index")

    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    if column_rename_map is not None:
        for source_col, target_col in column_rename_map.items():
            optional_columns[target_col] = predictions[source_col].values

    # Kill child processes spawned by Joblib to avoid spam in the AMLB log
    get_reusable_executor().shutdown(wait=True)

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions_only,
        truth=truth_only,
        target_is_encoded=False,
        models_count=1,
        training_duration=training_time,
        predict_duration=predict_time,
        optional_columns=pd.DataFrame(optional_columns),
    )


def get_probabilistic_forecast_config(quantile_levels):
    if len(quantile_levels) <= 3:
        q_min = min(quantile_levels)
        q_max = max(quantile_levels)
        point_alpha = sorted(quantile_levels)[1]
        coverage = [q_min, q_max]
        column_rename_map = {
            "lower": str(q_min),
            "y_pred": str(point_alpha),
            "upper": str(q_max),
        }
    else:
        log.warning("PyCaret supports at most 3 quantile levels, falling back to point forecast")
        coverage = None
        point_alpha = None
        column_rename_map = None
    return coverage, point_alpha, column_rename_map


def get_eval_metric(metric: str) -> dict:
    if metric in ['sql', 'mase']:
        return 'MASE'
    elif metric in ['rmse', 'mse']:
        return 'RMSE'
    elif metric in ['mae', 'wql', 'mql']:
        return 'MAE'
    elif metric in ['mape']:
        return 'MAPE'
    elif metric in ['smape']:
        return 'SMAPE'
    else:
        log.warning(f"Metric {metric} not supported, falling back to 'MASE'")
        return 'MASE'


if __name__ == '__main__':
    call_run(run)
