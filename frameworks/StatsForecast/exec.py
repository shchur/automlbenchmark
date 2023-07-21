import time
import logging

log = logging.getLogger(__name__)

log.warning("SLEEPING FOR 10000")
time.sleep(10000)

import numpy as np
import pandas as pd
import warnings

warnings.simplefilter('ignore')

from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    Naive,
    SeasonalNaive,
)

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer


def run(dataset, config):
    train_data = pd.read_csv(dataset.train_path)
    train_data.rename(
        columns={
            dataset.id_column: 'unique_id',
            dataset.timestamp_column: 'ds',
            dataset.target: 'y',
        },
        inplace=True,
    )

    models = get_models(
        framework_params=config.framework_params,
        seasonality=7 if dataset.freq == "D" else dataset.seasonality,
    )
    model_name = repr(models[0])
    # Convert quantile_levels (floats in (0, 1)) to confidence levels (ints in [0, 100]) used by StatsForecast
    levels = []
    quantile_to_key = {}
    for q in config.quantile_levels:
        level = round(abs(q - 0.5) * 200, 1)
        suffix = 'lo' if q < 0.5 else 'hi'
        levels.append(level)
        quantile_to_key[str(q)] = f'{model_name}-{suffix}-{level}'
    levels = sorted(list(set(levels)))

    sf = StatsForecast(
        models=models,
        freq=dataset.freq,
        n_jobs=config.cores,
        fallback_model=SeasonalNaive(season_length=dataset.seasonality),
    )
    with Timer() as predict:
        predictions = sf.forecast(
            df=train_data, h=dataset.forecast_horizon_in_steps, level=levels
        )

    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in config.quantile_levels:
        optional_columns[str(q)] = predictions[quantile_to_key[str(q)]].values

    predictions_only = predictions[model_name].values
    test_data_future = pd.read_csv(dataset.test_path)
    truth_only = test_data_future[dataset.target].values

    # Sanity check - make sure predictions are ordered correctly
    if (predictions.index != test_data_future[dataset.id_column]).any():
        raise AssertionError(
            "item_id column for predictions doesn't match test data index"
        )

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions_only,
        truth=truth_only,
        target_is_encoded=False,
        models_count=1,
        training_duration=0.0,
        predict_duration=predict.duration,
        optional_columns=pd.DataFrame(optional_columns),
    )


def get_models(framework_params: dict, seasonality: int):
    model_name = framework_params.get('model_name', 'SeasonalNaive').lower()
    extra_params = {
        k: v
        for k, v in framework_params.items()
        if not (k.startswith('_') or k == 'model_name')
    }
    if model_name == 'naive':
        return [Naive()]
    elif model_name == 'seasonalnaive':
        return [SeasonalNaive(season_length=seasonality)]
    elif model_name == 'autoarima':
        return [AutoARIMA(season_length=seasonality, **extra_params)]
    elif model_name == 'autoets':
        return [AutoETS(season_length=seasonality, **extra_params)]
    elif model_name == 'autotheta':
        return [AutoTheta(season_length=seasonality, **extra_params)]
    else:
        raise ValueError(f'Unsupported model name {model_name}')


if __name__ == '__main__':
    call_run(run)
