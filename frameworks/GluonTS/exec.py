import logging
import numpy as np
import pandas as pd
import warnings
from datetime import timedelta

warnings.simplefilter("ignore")

import gluonts
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from pytorch_lightning.callbacks import Timer as TimerCallback
from pytorch_lightning.callbacks import EarlyStopping

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


default_params = {
    "max_epochs": 5000,
    "patience": 50,
    "model_name": "DeepAR",
    "batch_size": 64,
}


def run(dataset, config):
    train_df = pd.read_csv(dataset.train_path, parse_dates=[dataset.timestamp_column])
    train_data, val_data = train_val_split(
        train_df,
        prediction_length=dataset.forecast_horizon_in_steps,
        id_column=dataset.id_column,
        timestamp_column=dataset.timestamp_column,
        target_column=dataset.target,
    )

    params = default_params.copy()
    for k, v in config.framework_params.items():
        if not k.startswith('_'):
            params[k] = v

    estimator_cls = get_estimator_class(params.pop("model_name").lower())

    gts_logger = logging.getLogger(gluonts.__name__)
    gts_logger.setLevel(logging.ERROR)
    callbacks = [
        TimerCallback(timedelta(seconds=config.max_runtime_seconds)),
        EarlyStopping(monitor="val_loss", patience=params.pop("patience"))
    ]
    estimator = estimator_cls(
        freq=dataset.freq,
        prediction_length=dataset.forecast_horizon_in_steps,
        trainer_kwargs={
            "max_epochs": params.pop("max_epochs"),
            "enable_progress_bar": False,
            "accelerator": "cpu",
            "callbacks": callbacks,
        },
        **params,
    )

    with Timer() as training:
        predictor = estimator.train(training_data=train_data, validation_data=val_data, cache_data=True)

    with Timer() as predict:
        forecasts = list(predictor.predict(val_data))

    predictions = forecasts_to_data_frame(forecasts, config.quantile_levels)
    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in config.quantile_levels:
        optional_columns[str(q)] = predictions[str(q)].values

    predictions_only = get_point_forecast(predictions, config.metric)
    test_data_future = pd.read_csv(dataset.test_path)
    truth_only = test_data_future[dataset.target].values

    # Sanity check - make sure predictions are ordered correctly
    if (predictions["item_id"] != test_data_future[dataset.id_column]).any():
        raise AssertionError(
            "item_id column for predictions doesn't match test data index"
        )

    if (predictions["timestamp"] != test_data_future[dataset.timestamp_column]).any():
        log.info(predictions["timestamp"])
        log.info(test_data_future[dataset.timestamp_column])
        raise AssertionError(
            "timestamp column for predictions doesn't match test data index"
        )

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


def get_estimator_class(model_name: str):
    if model_name == "deepar":
        return DeepAREstimator
    elif model_name == "tft":
        return TemporalFusionTransformerEstimator
    else:
        raise ValueError(f"Unsupported model name {model_name}")


def train_val_split(full_data, prediction_length, id_column, timestamp_column, target_column):
    if full_data.groupby(id_column, sort=False).size().min() <= prediction_length:
        raise ValueError("Time series too short to generate validation set")
    train_data = full_data.groupby(id_column, sort=False, as_index=False).nth(slice(None, -prediction_length))
    train_data = to_gluonts_dataset(train_data, id_column, timestamp_column, target_column)
    val_data = to_gluonts_dataset(full_data, id_column, timestamp_column, target_column)
    return train_data, val_data


def to_gluonts_dataset(df, id_column, timestamp_column, target_column):
    df = df.rename(columns={target_column: "target"})
    return PandasDataset.from_long_dataframe(df, item_id=id_column, timestamp=timestamp_column)


def forecasts_to_data_frame(forecasts, quantile_levels) -> pd.DataFrame:
    dfs = []
    for f in forecasts:
        forecast_dict = {"mean": f.mean}
        for q in quantile_levels:
            forecast_dict[str(q)] = f.quantile(q)
        df = pd.DataFrame(forecast_dict)
        df["item_id"] = f.item_id
        df["timestamp"] = pd.date_range(start=f.start_date.to_timestamp("S"), freq=f.start_date.freq, periods=len(df))
        dfs.append(df)
    return pd.concat(dfs, axis=0).reset_index(drop=True)


def get_point_forecast(predictions, metric):
    # Return median for metrics optimized by median, if possible
    if metric.lower() in ["rmse", "mse"] or "0.5" not in predictions.columns:
        log.info("Using mean as point forecast")
        return predictions["mean"].values
    else:
        log.info("Using median as point forecast")
        return predictions["0.5"].values


if __name__ == "__main__":
    call_run(run)
