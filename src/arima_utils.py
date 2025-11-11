import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


def get_arima_dataset(dataset, station: str, parameter: str = "temperature"):
    train = dataset.get_observations(
        stations=station,
        parameters=parameter,
        first_date="2020-01-01",
        last_date="2020-12-31",
    )
    test = dataset.get_observations(
        stations=station,
        parameters=parameter,
        first_date="2021-01-01",
        last_date="2021-01-31",
    )
    return train, test


def plot_pandas_series(series):
    series.plot(
        title="Time series plot",
        ylabel="Value",
        xlabel="Date",
        figsize=(12, 6),
        legend=False,
    )
    plt.show()


def pacf_plot(series, lags=48):
    plt.figure(figsize=(12, 4))
    plot_pacf(series, ax=plt.gca(), lags=lags, method="ywm")
    plt.title("PACF (partial auto-correlation)")
    plt.show()


def test_arima_model(model, h: int, series):
    w = max(
        *model.model.order,
        *(model.model.seasonal_order[3] * o for o in model.model.seasonal_order[:3]),
    )
    maes = []
    results = []
    for i in range(w, len(series) - h):
        window = series[i - w : i]
        horizon = series[i : i + h]
        # Create a new SARIMA model with the same structure
        forecast = (
            SARIMAX(
                endog=window,
                order=model.model.order,
                seasonal_order=model.model.seasonal_order,
            )
            .filter(model.params)
            .get_forecast(steps=h)
            .predicted_mean
        )
        mae = np.abs(horizon.values.squeeze(-1) - forecast.values).mean()
        maes.append(mae)
        results.append({"window": window, "horizon": horizon, "forecast": forecast})
    return np.mean(maes), results


def plot_results(results, num_samples=5):
    for _ in range(num_samples):
        i = np.random.randint(0, len(results))
        sample = results[i]
        plt.figure(figsize=(12, 4))
        plt.plot(sample["window"].index, sample["window"], label="input window")
        plt.plot(sample["horizon"].index, sample["horizon"], label="true horizon")
        plt.plot(sample["forecast"].index, sample["forecast"], label="forecast")
        plt.legend()
        plt.title("SARIMA Forecast vs True Horizon")
        plt.show()
