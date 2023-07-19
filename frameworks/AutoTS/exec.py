import contextlib
import logging
import os
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter('ignore')

from autots import AutoTS
from joblib.externals.loky import get_reusable_executor

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


def run(dataset, config):
    train_data = pd.read_csv(dataset.train_path, parse_dates=[dataset.timestamp_column])

    prediction_interval = compute_prediction_interval(config.quantile_levels)
    framework_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    framework_params.setdefault("num_validations", 2)

    model = AutoTS(
        forecast_length=dataset.forecast_horizon_in_steps,
        frequency=dataset.freq,
        prediction_interval=prediction_interval if prediction_interval is not None else 0.9,
        verbose=0,
        generation_timeout=config.max_runtime_seconds // 60,
        metric_weighting=get_metric_weighting(config.metric),
        **framework_params,
    )

    with joblib_warning_filter():
        with Timer() as training:
            model = model.fit(
                train_data,
                date_col=dataset.timestamp_column,
                value_col=dataset.target,
                id_col=dataset.id_column,
            )

        with Timer() as predict:
            predictions = model.predict()

    point_forecast = convert_forecast_to_long_format(predictions.forecast)

    predictions_only = point_forecast.values
    test_data_future = pd.read_csv(dataset.test_path)
    truth_only = test_data_future[dataset.target].values

    # Sanity check - make sure predictions are ordered correctly
    if (point_forecast.index.get_level_values(0) != test_data_future[dataset.id_column]).any():
        raise AssertionError("item_id column for predictions doesn't match test data index")

    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    if prediction_interval is not None:
        for q in config.quantile_levels:
            if q == 0.5:
                q_forecast = predictions.forecast
            elif q == min(config.quantile_levels):
                q_forecast = predictions.lower_forecast
            elif q == max(config.quantile_levels):
                q_forecast = predictions.upper_forecast
            else:
                raise ValueError("This should never happen")
            optional_columns[str(q)] = convert_forecast_to_long_format(q_forecast).values

    # Kill child processes spawned by Joblib to avoid spam in the AMLB log
    get_reusable_executor().shutdown(wait=True)

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


def compute_prediction_interval(quantile_levels):
    # AutoTS only supports 2 quantile levels (except median) that are symmetric about P50.
    # Return the corresponding prediction_interval if quantile_levels are supported, otherwise return None
    quantile_levels = [q for q in quantile_levels if q != 0.5]
    if len(quantile_levels) == 2 and np.allclose(quantile_levels[0], 1.0 - quantile_levels[1]):
        return 1 - 0.5 * min(quantile_levels)
    else:
        return None


def convert_forecast_to_long_format(forecast: pd.DataFrame) -> pd.DataFrame:
    forecast = forecast.stack().swaplevel().sort_index()
    forecast.index.rename(["item_id", "timestamp"], inplace=True)
    return forecast


@contextlib.contextmanager
def joblib_warning_filter():
    env_py_warnings = os.environ.get("PYTHONWARNINGS", "")
    try:
        os.environ["PYTHONWARNINGS"] = "ignore"
        yield
    finally:
        os.environ["PYTHONWARNINGS"] = env_py_warnings


def get_metric_weighting(metric: str) -> dict:
    if metric in ['spl', 'wql', 'mql']:
        return {'spl_weighting': 100}
    elif metric in ['rmse', 'mse']:
        return {'rmse_weighting': 100}
    elif metric in ['mae', 'mase']:
        return {'mae_weighting': 100}
    elif metric in ['mape', 'smape']:
        return {'smape_weighting': 100}
    else:
        # Default metric_weighting
        return {
            'smape_weighting': 5,
            'mae_weighting': 2,
            'rmse_weighting': 2,
            'made_weighting': 0.5,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0,
            'spl_weighting': 3,
            'containment_weighting': 0,
            'contour_weighting': 1,
            'runtime_weighting': 0.05,
            'oda_weighting': 0.001,
        }


if __name__ == '__main__':
    call_run(run)
