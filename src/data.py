from collections import namedtuple

import matplotlib.pyplot as plt
import torch
from peakweather.dataset import PeakWeatherDataset
from torch.utils.data import Dataset, DataLoader
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx


def shift_date_back(date: str, hours: int) -> str:
    from datetime import datetime, timedelta

    dt = datetime.strptime(date, "%Y-%m-%d")
    dt_shifted = dt - timedelta(hours=hours)
    return dt_shifted.strftime("%Y-%m-%d %H:%M:%S")


class PeakWeatherTorchDataset(Dataset):
    Sample = namedtuple("Sample", ["x", "y", "mu", "sigma"])

    def __init__(self, window: int, horizon: int, parameter: str = "temperature"):
        ds = PeakWeatherDataset(
            root=None,
            compute_uv=False,
            station_type="meteo_station",
            freq="h",
            aggregation_methods={"temperature": "mean"},
        )
        self.mode = "train"
        self.window = window
        self.horizon = horizon
        train, mask = ds.get_observations(
            parameters=parameter,
            first_date="2020-01-01",
            last_date="2020-11-30",
            as_numpy=True,
            return_mask=True,
        )
        good_stations = (mask.sum(axis=0) > 0).squeeze()
        self.data = {
            "train": train[:, good_stations].squeeze(),
            "val": ds.get_observations(
                parameters=parameter,
                first_date=shift_date_back("2020-12-01", hours=self.window),
                last_date="2020-12-31",
                as_numpy=True,
            )[:, good_stations].squeeze(),
            "test": ds.get_observations(
                parameters=parameter,
                first_date=shift_date_back("2021-01-01", hours=self.window),
                last_date="2021-01-31",
                as_numpy=True,
            )[:, good_stations].squeeze(),
        }
        self.scaling_params = {
            "mu": self.data["train"].mean(axis=0),
            "sigma": self.data["train"].std(axis=0),
        }
        for mode in ["train", "val", "test"]:
            self.data[mode] = (
                self.data[mode] - self.scaling_params["mu"]
            ) / self.scaling_params["sigma"]

    def __len__(self):
        return (self.data[self.mode].shape[0] - self.window - self.horizon) * self.data[
            self.mode
        ].shape[1]

    def __getitem__(self, idx):
        n, t = divmod(idx, (self.data[self.mode].shape[0] - self.window - self.horizon))
        x = self.data[self.mode][t : t + self.window, n]
        y = self.data[self.mode][t + self.window : t + self.window + self.horizon, n]
        mu = self.scaling_params["mu"][n]
        sigma = self.scaling_params["sigma"][n]
        return self.Sample(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            mu=torch.tensor(mu, dtype=torch.float32),
            sigma=torch.tensor(sigma, dtype=torch.float32),
        )


def test_model(model: torch.nn.Module, test_loader: DataLoader) -> float:
    criterion = torch.nn.L1Loss()
    model.eval()
    total_test = 0
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch.x)
            preds_rescaled = preds * batch.sigma.unsqueeze(-1) + batch.mu.unsqueeze(-1)
            y_rescaled = batch.y * batch.sigma.unsqueeze(-1) + batch.mu.unsqueeze(-1)
            loss = criterion(preds_rescaled, y_rescaled)
            total_test += loss.item()
    avg_test = total_test / len(test_loader)
    print(f"Test MAE: {avg_test:.4f}")
    return avg_test


def plot_predictions(
    model: torch.nn.Module, test_dataset: Dataset, num_samples: int = 5
):
    model.eval()
    fig, axs = plt.subplots(num_samples, 1, figsize=(10, num_samples * 3))
    with torch.no_grad():
        for i in range(num_samples):
            n = torch.randint(0, len(test_dataset), (1,)).item()
            sample = test_dataset[n]
            preds = model(sample.x.unsqueeze(0)).squeeze(0).numpy()
            # Rescale predictions and ground truth
            preds_rescaled = preds * sample.sigma.numpy() + sample.mu.numpy()
            y_rescaled = sample.y.numpy() * sample.sigma.numpy() + sample.mu.numpy()
            axs[i].plot(
                range(len(sample.x)),
                sample.x.numpy() * sample.sigma.numpy() + sample.mu.numpy(),
                label="input window",
            )
            axs[i].plot(
                range(len(sample.x), len(sample.x) + len(y_rescaled)),
                y_rescaled,
                label="true horizon",
            )
            axs[i].plot(
                range(len(sample.x), len(sample.x) + len(preds_rescaled)),
                preds_rescaled,
                label="forecast",
            )
            axs[i].legend()
            axs[i].set_title(f"Sample {n}")
    plt.tight_layout()
    plt.show()


def plot_stations_on_map(dataset, station_ids: list[str] | None = None):
    # Get stations data
    df = (
        dataset.stations_table[["latitude", "longitude"]]
        if station_ids is None
        else dataset.stations_table.loc[station_ids, ["latitude", "longitude"]]
    )
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)],
        crs="EPSG:4326",
    )

    # --- Load Swiss canton boundaries from GeoBoundaries ---
    cantons = gpd.read_file(
        "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/CHE/ADM1/geoBoundaries-CHE-ADM1_simplified.geojson"
    )
    # Create the national border by dissolving cantons
    swiss_border = cantons.dissolve()

    # --- Project to Web Mercator (for basemap compatibility) ---
    gdf_web = gdf.to_crs(epsg=3857)
    border_web = swiss_border.to_crs(epsg=3857)
    cantons_web = cantons.to_crs(epsg=3857)

    # --- Plot everything ---
    fig, ax = plt.subplots(figsize=(12, 12))
    border_web.boundary.plot(ax=ax, color="darkred", linewidth=1.5)
    cantons_web.boundary.plot(ax=ax, color="goldenrod", linewidth=0.6)
    gdf_web.plot(ax=ax, color="red", markersize=25, zorder=5)

    # Add basemap (OpenStreetMap)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Add labels
    for x, y, label in zip(gdf_web.geometry.x, gdf_web.geometry.y, gdf_web.index):
        ax.text(x + 5000, y, label, fontsize=6, color="black", weight="bold")

    ax.set_title("Selected meteorological stations in Switzerland", fontsize=14)
    ax.set_axis_off()
    plt.show()


def plot_timeseries(
    dataset, station_ids: list[str], parameter: str, start_date: str, end_date: str
):
    station = dataset.get_observations(
        stations=station_ids,
        parameters=parameter,
        first_date=start_date,
        last_date=end_date,
    )
    station.plot(
        ylabel=parameter.capitalize(),
        xlabel="Date",
        title="Time series at selected meteorological stations in Switzerland",
        legend=True,
        subplots=True,
        figsize=(10, 6),
        sharex=True,
        sharey=True,
    )
    plt.show()
