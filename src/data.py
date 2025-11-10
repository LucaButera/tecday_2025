from collections import namedtuple

import matplotlib.pyplot as plt
import torch
from peakweather.dataset import PeakWeatherDataset
from torch.utils.data import Dataset, DataLoader


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
                first_date="2020-12-01",
                last_date="2020-12-31",
                as_numpy=True,
            )[:, good_stations].squeeze(),
            "test": ds.get_observations(
                parameters=parameter,
                first_date="2021-01-01",
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
