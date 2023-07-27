import logging
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter('ignore')

from neuralprophet import NeuralProphet

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)


def run(dataset, config):
    train_data = rename_columns(pd.read_csv(dataset.train_path), dataset)
    train_data['ID'] = train_data['ID'].astype("str")

    model = NeuralProphet(quantiles=config.quantile_levels)
    # Suppress info messages
    np_logger = logging.getLogger("NP.df_utils")
    np_logger.setLevel(logging.ERROR)

    with Timer() as training:
        model.fit(train_data, freq=dataset.freq, progress=False)

    test_data_future = rename_columns(pd.read_csv(dataset.test_path), dataset)
    truth_only = test_data_future['y'].values.copy()
    # Hide target values before forecast
    test_data_future['y'] = np.nan

    with Timer() as predict:
        predictions = model.predict(test_data_future)

    predictions_only = predictions["yhat1"].values

    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in config.quantile_levels:
        if str(q) == "0.5":
            col_name = "yhat1"
        else:
            col_name = f"yhat1 {q:.1%}"
        optional_columns[str(q)] = predictions[col_name].values

    # Sanity check - make sure predictions are ordered correctly
    if (predictions['ID'] != test_data_future['ID'].astype('str')).any():
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


def rename_columns(df, dataset):
    return df.rename(columns={dataset.id_column: 'ID', dataset.timestamp_column: 'ds', dataset.target: 'y'})


if __name__ == '__main__':
    call_run(run)
