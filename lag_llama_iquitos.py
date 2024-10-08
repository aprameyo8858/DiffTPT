# -*- coding: utf-8 -*-
"""Lag_Llama_Iquitos.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FBeoiNwBLaA8OWtARCGk3kF1K7RIudCw
"""

!git clone https://github.com/time-series-foundation-models/lag-llama/

cd /content/lag-llama

!pip install -r requirements.txt --quiet

!huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir /content/lag-llama

from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from tqdm.autonotebook import tqdm

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

from lag_llama.gluon.estimator import LagLlamaEstimator

def get_lag_llama_predictions(dataset, prediction_length, context_length=32, num_samples=20, device="cpu", batch_size=64, nonnegative_pred_samples=True):
    ckpt = torch.load("lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,
        device = torch.device('cpu'),
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],

        nonnegative_pred_samples=nonnegative_pred_samples,

        # linear positional encoding scaling
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        },

        batch_size=batch_size,
        num_parallel_samples=num_samples,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

    return forecasts, tss

import pandas as pd
from datetime import datetime, timedelta

# Sample data
data = pd.read_excel('/content/Iquitos_Dengue(1).xlsx')

# Create DataFrame
df = pd.DataFrame(data)

df['Year'] = df['Year'].astype(int)
df['Week'] = df['Week'].astype(int)
print(df.dtypes)

# Function to convert Year and Week to the specified date format
def convert_to_date(year, week):
    # Find the first day of the given year
    year = int(year)
    first_day_of_year = datetime(year, 1, 1)
    # Adjust if the first day of the year is not a Monday
    days_to_monday = (7 - first_day_of_year.weekday()) % 7
    first_monday = first_day_of_year + timedelta(days=days_to_monday)
    # Calculate the target date
    target_date = first_monday + timedelta(weeks=week - 1)
    return target_date.strftime('%m-%d-%Y 00:00:00')

# Apply the conversion function to each row
df['date'] = df.apply(lambda row: convert_to_date(row['Year'], row['Week']), axis=1)

print(df)

df.head(60)

df.set_index('date', inplace=True)

df

df.head(60)

df = df.drop('Week',axis=1)
df = df.drop('Year',axis=1)
df = df.drop('Rain', axis=1)

df_train = df.iloc[:520]
df_test = df.iloc[520:572]

df_test

print(df_train)

from gluonts.dataset.common import ListDataset

# Assuming 'total' is the target time series
training_data = ListDataset(
    [{"start": df_train.index[0], "target": df_train["Cases"].values}],
    freq="W"
)

from gluonts.dataset.common import ListDataset

# Assuming 'total' is the target time series
testing_data = ListDataset(
    [{"start": df_test.index[0], "target": df_test["Cases"].values}],
    freq="W"
)

prediction_length = 10
context_length = prediction_length*3
num_samples = 20
device = "cpu"

forecasts, tss = get_lag_llama_predictions(
    testing_data,
    prediction_length=prediction_length,
    num_samples=num_samples,
    context_length=context_length,
    device=device
)

plt.figure(figsize=(20, 15))
date_formater = mdates.DateFormatter('%b, %d')
plt.rcParams.update({'font.size': 15})

# Iterate through the first 9 series, and plot the predicted samples
for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
    ax = plt.subplot(3, 3, idx+1)

    plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target", )
    forecast.plot( color='g')
    plt.xticks(rotation=60)
    ax.xaxis.set_major_formatter(date_formater)
    ax.set_title(forecast.item_id)

plt.gcf().tight_layout()
plt.legend()
plt.show()

evaluator = Evaluator()
agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
agg_metrics

ckpt = torch.load("lag-llama.ckpt", map_location=device)
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,
        device = torch.device('cpu'),

        # distr_output="neg_bin",
        # scaling="mean",
        nonnegative_pred_samples=True,
        aug_prob=0,
        lr=5e-4,

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        time_feat=estimator_args["time_feat"],

        # rope_scaling={
        #     "type": "linear",
        #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        # },

        batch_size=64,
        num_parallel_samples=num_samples,
        trainer_kwargs = {"max_epochs": 50,}, # <- lightning trainer arguments
    )

predictor = estimator.train(training_data, cache_data=True, shuffle_buffer_length=1000)

forecast_it, ts_it = make_evaluation_predictions(
        dataset=testing_data,
        predictor=predictor,
        num_samples=num_samples
    )

forecasts = list(tqdm(forecast_it, total=len(testing_data), desc="Forecasting batches"))

tss = list(tqdm(ts_it, total=len(testing_data), desc="Ground truth"))

plt.figure(figsize=(20, 15))
date_formater = mdates.DateFormatter('%b, %d')
plt.rcParams.update({'font.size': 15})

# Iterate through the first 9 series, and plot the predicted samples
for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
    ax = plt.subplot(3, 3, idx+1)

    plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target", )
    forecast.plot( color='g')
    plt.xticks(rotation=60)
    ax.xaxis.set_major_formatter(date_formater)
    ax.set_title(forecast.item_id)

plt.gcf().tight_layout()
plt.legend()
plt.show()

evaluator = Evaluator()
agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
agg_metrics

