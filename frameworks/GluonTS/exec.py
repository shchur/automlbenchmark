import logging
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter("ignore")

import gluonts
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.model.seasonal_naive import SeasonalNaivePredictor

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


def run(dataset, config):
    train_df = pd.read_csv(dataset.train_path, parse_dates=[dataset.timestamp_column])
    train_data, val_data, full_data = train_val_split(
        train_df,
        freq=dataset.freq,
        prediction_length=dataset.forecast_horizon_in_steps,
        id_column=dataset.id_column,
        timestamp_column=dataset.timestamp_column,
        target_column=dataset.target,
    )

    estimator_cls = get_estimator_class(
        config.framework_params.get("model_name", "DeepAR").lower()
    )

    gts_logger = logging.getLogger(gluonts.__name__)
    pl_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if "lightning" in name]
    for logger in pl_loggers:
        logger.setLevel(logging.ERROR)
    gts_logger.setLevel(logging.ERROR)
    estimator = estimator_cls(
        freq=dataset.freq,
        prediction_length=dataset.forecast_horizon_in_steps,
        trainer_kwargs={"max_epochs": 2, "enable_progress_bar": False},
    )

    # with Timer() as training:
        # predictor = SeasonalNaivePredictor(freq=dataset.freq, prediction_length=dataset.forecast_horizon_in_steps, season_length=dataset.seasonality)
    # with gluonts.core.settings.let(gluonts.env.env, use_tqdm=False):
    predictor = estimator.train(training_data=train_data, validation_data=None, cache_data=True)

    # with Timer() as predict:
    forecasts = predictor.predict(full_data)

    predictions = forecasts_to_data_frame(forecasts, config.quantile_levels)
    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in config.quantile_levels:
        optional_columns[str(q)] = predictions[str(q)].values

    predictions_only = predictions["mean"].values
    test_data_future = pd.read_csv(dataset.test_path)
    truth_only = test_data_future[dataset.target].values

    print(f"PREDICTIONS {predictions['item_id']}")
    print(f"FUTURE {test_data_future[dataset.id_column]}")

    # Sanity check - make sure predictions are ordered correctly
    if (predictions["item_id"] != test_data_future[dataset.id_column]).any():
        raise AssertionError(
            "item_id column for predictions doesn't match test data index"
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


def train_val_split(full_data, prediction_length, freq, id_column, timestamp_column, target_column):
    if full_data.groupby(id_column, sort=False).size().min() > prediction_length:
        log.warning(f"Using last {prediction_length} values of each time series as a validation set")
        train_data = full_data.groupby(id_column, sort=False, as_index=False).nth(slice(None, -prediction_length))
        val_data = full_data
    else:
        log.warning("Provided data is too short to generate a validation set, disabling validation")
        train_data = full_data
        val_data = None
    train_data = to_gluonts_dataset(train_data, freq, id_column, timestamp_column, target_column)
    val_data = to_gluonts_dataset(val_data, freq, id_column, timestamp_column, target_column)
    full_data = to_gluonts_dataset(full_data, freq, id_column, timestamp_column, target_column)
    return train_data, val_data, full_data


def to_gluonts_dataset(df, freq, id_column, timestamp_column, target_column):
    if df is None:
        return None
    else:
        entries = []
        for item_id, ts in df.groupby(id_column, sort=False):
            entries.append(
                {
                    "target": ts[target_column],
                    "start": pd.Period(ts[timestamp_column].iloc[0], freq=freq),
                    "item_id": item_id,
                }
            )
        return entries


def forecasts_to_data_frame(forecasts, quantile_levels) -> pd.DataFrame:
    dfs = []
    for f in forecasts:
        forecast_dict = {"mean": f.mean}
        for q in quantile_levels:
            forecast_dict[str(q)] = f.quantile(q)
        df = pd.DataFrame(forecast_dict)
        df["item_id"] = f.item_id
        dfs.append(df)
    return pd.concat(dfs, axis=0).reset_index(drop=True)



if __name__ == "__main__":
    call_run(run)
