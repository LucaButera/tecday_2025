import math
from functools import partial

import torch
from einops import rearrange, repeat, reduce
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float, Bool, Int
from torch import Tensor
from torch.utils.data import DataLoader
from uni2ts.common.torch_util import packed_causal_attention_mask
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from uni2ts.module.transformer import TransformerEncoder
from uni2ts.module.ts_embed import ResidualBlock
import torch.nn.functional as F


class Moirai2Module(
    torch.nn.Module,
    PyTorchModelHubMixin,
):
    """
    Contains components of Moirai, to ensure implementation is identical across models.
    Subclasses huggingface_hub.PyTorchModelHubMixin to support loading from HuggingFace Hub.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_layers: int,
        patch_size: int,
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        num_predict_token: int = 1,
        quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ):
        """
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_size: patch size
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param num_quantiles: number of quantile levels
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_predict_token = num_predict_token
        self.max_seq_len = max_seq_len
        self.quantile_levels = quantile_levels
        self.num_quantiles = len(quantile_levels)

        self.in_proj = ResidualBlock(
            input_dims=patch_size * 2,
            hidden_dims=d_model,
            output_dims=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=d_ff,
        )
        self.out_proj = ResidualBlock(
            input_dims=d_model,
            hidden_dims=d_model,
            output_dims=num_predict_token * self.num_quantiles * patch_size,
        )

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ):
        """
        Defines the forward pass of MoiraiDecoderModule.
        This method expects processed inputs.

        1. Apply scaling to observations
        2. Project from observations to representations
        3. Replace prediction window with learnable mask
        4. Apply transformer layers
        5. Project from representations to distribution parameters
        6. Return distribution object

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :return: predictive distribution
        """
        input_tokens = torch.cat([target, observed_mask.to(torch.float32)], dim=-1)
        reprs = self.in_proj(input_tokens)

        reprs = self.encoder(
            reprs,
            packed_causal_attention_mask(sample_id, time_id),
            time_id=time_id,
            var_id=variate_id,
        )
        preds = self.out_proj(reprs)
        return preds


class MoiraiModel(torch.nn.Module):
    def __init__(
        self,
        horizon: int,
        window: int,
    ):
        super().__init__()
        self.window = window
        self.horizon = horizon
        self.module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")

    def context_token_length(self, patch_size: int) -> int:
        return math.ceil(self.window / patch_size)

    def prediction_token_length(self, patch_size) -> int:
        return math.ceil(self.horizon / patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        (
            patched_x,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
        ) = self._preprocess_input(
            self.module.patch_size,
            x,
            past_observed_target=~torch.isnan(x),
            past_is_pad=torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device),
        )

        preds = self.module(
            patched_x,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
        )

        outputs = self._postprocess_output(patched_x, preds).squeeze(-1)
        return outputs

    def _postprocess_output(
        self,
        patched_x: Tensor,
        preds: Tensor,
    ) -> Tensor:
        per_var_context_token = self.context_token_length(self.module.patch_size)
        total_context_token = per_var_context_token
        per_var_predict_token = self.prediction_token_length(self.module.patch_size)
        total_predict_token = per_var_predict_token

        pred_index = torch.arange(
            start=per_var_context_token - 1,
            end=total_context_token,
            step=per_var_context_token,
        )
        assign_index = torch.arange(
            start=total_context_token,
            end=total_context_token + total_predict_token,
            step=per_var_predict_token,
        )
        quantile_prediction = repeat(
            patched_x,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=len(self.module.quantile_levels),
            patch_size=self.module.patch_size,
        ).clone()

        assert per_var_predict_token <= self.module.num_predict_token, (
            "Recursive prediction is not supported"
        )
        preds, adjusted_assign_index = self._structure_multi_predict(
            per_var_predict_token,
            pred_index,
            assign_index,
            preds,
        )
        quantile_prediction[..., adjusted_assign_index, :, :] = preds
        quantile_prediction = self._format_preds(
            self.module.num_quantiles,
            self.module.patch_size,
            quantile_prediction,
            1,
        )
        quantile_prediction = torch.median(quantile_prediction, dim=1).values
        return quantile_prediction

    def _structure_multi_predict(
        self,
        per_var_predict_token,
        pred_index,
        assign_index,
        preds,
    ):
        preds = rearrange(
            preds,
            "... (predict_token num_quantiles patch_size) -> ... predict_token num_quantiles patch_size",
            predict_token=self.module.num_predict_token,
            num_quantiles=self.module.num_quantiles,
            patch_size=self.module.patch_size,
        )
        preds = rearrange(
            preds[..., pred_index, :per_var_predict_token, :, :],
            "... pred_index predict_token num_quantiles patch_size -> ... (pred_index predict_token) num_quantiles patch_size",
        )
        adjusted_assign_index = torch.cat(
            [
                torch.arange(start=idx, end=idx + per_var_predict_token)
                for idx in assign_index
            ]
        )
        return preds, adjusted_assign_index

    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: float | None = None,
    ) -> torch.Tensor:
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _generate_time_id(
        self,
        patch_size: int,
        past_observed_target: torch.Tensor,  # "batch past_seq tgt" | bool
    ) -> tuple[
        torch.Tensor,  # "batch past_token" | int
        torch.Tensor,  # "batch future_token" | int
    ]:
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(
            past_seq_id.cummax(dim=-1).values.cumsum(dim=-1) - 1, min=0
        )
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    self.prediction_token_length(patch_size),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        return past_seq_id, future_seq_id

    def _preprocess_input(
        self,
        patch_size: int,
        past_target: torch.Tensor,  # "batch past_time tgt" | float
        past_observed_target: torch.Tensor,  # "batch past_time tgt" | bool
        past_is_pad: torch.Tensor,  # "batch past_time" | bool
    ) -> tuple[
        torch.Tensor,  # "batch combine_seq patch" | float # target
        torch.Tensor,  # "batch combine_seq patch" | bool  # observed_mask
        torch.Tensor,  # "batch combine_seq" | int         # sample_id
        torch.Tensor,  # "batch combine_seq" | int         # time_id
        torch.Tensor,  # "batch combine_seq" | int         # variate_id
    ]:
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

        future_target = torch.zeros(
            batch_shape
            + (
                self.horizon,
                past_target.shape[-1],
            ),
            dtype=past_target.dtype,
            device=device,
        )
        target.extend(
            [
                rearrange(
                    self._patched_seq_pad(patch_size, past_target, -2, left=True),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                rearrange(
                    self._patched_seq_pad(patch_size, future_target, -2, left=False),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
            ]
        )
        future_observed_target = torch.ones(
            batch_shape
            + (
                self.horizon,
                past_observed_target.shape[-1],
            ),
            dtype=torch.bool,
            device=device,
        )
        observed_mask.extend(
            [
                rearrange(
                    self._patched_seq_pad(
                        patch_size, past_observed_target, -2, left=True
                    ),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                rearrange(
                    self._patched_seq_pad(
                        patch_size, future_observed_target, -2, left=False
                    ),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
            ]
        )
        future_is_pad = torch.zeros(
            batch_shape + (self.horizon,),
            dtype=torch.long,
            device=device,
        )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device),
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device),
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
        )

    def _format_preds(
        self,
        num_quantiles: int,
        patch_size: int,
        preds: torch.Tensor,  # "batch combine_seq patch" | float
        target_dim: int,
    ) -> torch.Tensor:  # "batch num_quantiles future_time *tgt" | float
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :num_quantiles, :patch_size]
        preds = rearrange(
            preds,
            "... (dim seq) num_quantiles patch -> ... num_quantiles (seq patch) dim",
            dim=target_dim,
        )[..., : self.horizon, :]
        return preds


class GRUModel(torch.nn.Module):
    def __init__(
        self,
        horizon: int,
        hidden_size: int = 16,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=horizon)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)  # add feature dim
        _, h = self.gru(x)
        out = self.fc(h[-1].squeeze(0))
        return out


class MLPModel(torch.nn.Module):
    def __init__(
        self,
        window: int,
        horizon: int,
        hidden_size: int = 16,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                torch.nn.Linear(
                    in_features=window if i == 0 else hidden_size,
                    out_features=hidden_size,
                )
            )
            layers.append(torch.nn.ReLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(p=dropout))
        self.mlp = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=horizon)

    def forward(self, x: torch.Tensor):
        h = self.mlp(x)
        out = self.fc(h)
        return out


def train_model(
    model: torch.nn.Module,
    lr: float,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> torch.nn.Module:
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_train = 0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch.x)
            loss = criterion(preds, batch.y)
            loss.backward()
            optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)

        model.eval()
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch.x)
                loss = criterion(preds, batch.y)
                total_val += loss.item()
        avg_val = total_val / len(val_loader)

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()

        print(
            f"Epoch {epoch + 1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}"
        )

    # Restore best parameters
    model.load_state_dict(best_state)
    print(f"✅ Restored best model (val loss = {best_val_loss:.4f})")
    return model
