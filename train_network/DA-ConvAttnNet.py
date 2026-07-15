from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CyclicLR


TARGET_PARAM_NAMES = (
    "phig",
    "cit",
    "cdsc",
    "vsat",
    "dvt0",
    "dsub",
    "eta0",
    "atun",
    "btun",
    "phib",
)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    def _ensure_alias():
        sys.modules.setdefault("numpy._core", np)
        if hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    try:
        data = np.load(path, allow_pickle=False)
        _ = list(data.files)
    except Exception:
        _ensure_alias()
        data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_normalized_data(npz_path: Path) -> Dict:
    print(f"Loading normalized dataset: {npz_path}")
    data = load_npz(npz_path)

    n_vds_tr = int(data["n_vds_tr"][0])
    n_vgs_tr = int(data["n_vgs_tr"][0])
    n_tr = n_vds_tr * n_vgs_tr
    n_gm = int(data.get("n_gm_features", [n_tr])[0])

    print(f"  Transfer-current grid: {n_vds_tr} V_DS biases x {n_vgs_tr} V_GS points = {n_tr} features")
    print(f"  Transconductance features: {n_gm}")

    X_id_train_full = data["X_id_train"].astype(np.float32)
    X_id_val_full = data["X_id_val"].astype(np.float32)

    source_param_names = [str(x) for x in data["iv_param_names"].tolist()]
    source_param_lookup = {
        name.lower().replace("_", ""): idx for idx, name in enumerate(source_param_names)
    }
    missing_params = [name for name in TARGET_PARAM_NAMES if name not in source_param_lookup]
    if missing_params:
        raise ValueError(
            "The normalized dataset is missing the following paper45 target parameters: "
            + ", ".join(missing_params)
            + "; available parameters: "
            + ", ".join(source_param_names)
        )
    target_param_indices = np.asarray(
        [source_param_lookup[name] for name in TARGET_PARAM_NAMES], dtype=np.int64
    )
    source_is_log = data.get(
        "iv_param_is_log", np.zeros(len(source_param_names), dtype=bool)
    )

    X_T_train = X_id_train_full[:, :n_tr].reshape(-1, n_vds_tr, n_vgs_tr)
    X_gm_train = X_id_train_full[:, n_tr:n_tr + n_gm].reshape(-1, n_vds_tr, n_vgs_tr)
    X_T_val = X_id_val_full[:, :n_tr].reshape(-1, n_vds_tr, n_vgs_tr)
    X_gm_val = X_id_val_full[:, n_tr:n_tr + n_gm].reshape(-1, n_vds_tr, n_vgs_tr)

    result = {
        "X_T_train": X_T_train, "X_T_val": X_T_val,
        "X_gm_train": X_gm_train, "X_gm_val": X_gm_val,
        "X_geom_train": data["X_geom_train"].astype(np.float32),
        "X_geom_val": data["X_geom_val"].astype(np.float32),
        "y_train": data["y_train"][:, target_param_indices].astype(np.float32),
        "y_val": data["y_val"][:, target_param_indices].astype(np.float32),
        "y_train_raw": data["y_train_raw"][:, target_param_indices].astype(np.float32),
        "y_val_raw": data["y_val_raw"][:, target_param_indices].astype(np.float32),
        "iv_param_names": list(TARGET_PARAM_NAMES),
        "geom_param_names": [str(x) for x in data["geom_param_names"].tolist()],
        "n_vds_tr": n_vds_tr, "n_vgs_tr": n_vgs_tr,
        "scalers": {
            "id_mean": data["scaler_id_mean"], "id_std": data["scaler_id_std"],
            "geom_min": data["scaler_geom_min"], "geom_max": data["scaler_geom_max"],
            "iv_min": data["scaler_iv_min"][target_param_indices],
            "iv_max": data["scaler_iv_max"][target_param_indices],
            "iv_param_is_log": source_is_log[target_param_indices],
            "norm_low": float(data["norm_low"][0]),
            "norm_high": float(data["norm_high"][0]),
        },
        "vds_tr": data["vds_tr"], "vgs_tr": data["vgs_tr"],
    }

    print(f"  Training samples: {X_T_train.shape[0]}")
    print(f"  Validation samples: {X_T_val.shape[0]}")

    return result


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class AttentionPooling1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(channels))
        self.scale = channels ** -0.5
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        scores = (torch.tanh(x) * self.query.view(1, -1, 1)).sum(dim=1) * self.scale
        alpha = torch.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)
        return (x * alpha.unsqueeze(1)).sum(dim=-1)


class ResNetEncoder1D(nn.Module):


    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = [24, 48, 96],
        out_dim: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels[0], 3, padding=1),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.SiLU(),
        )
        blocks = []
        for i in range(len(hidden_channels) - 1):
            blocks.append(ResidualBlock1D(hidden_channels[i], hidden_channels[i + 1]))
        self.blocks = nn.Sequential(*blocks)
        self.pool = AttentionPooling1D(hidden_channels[-1], dropout)
        self.proj = nn.Linear(hidden_channels[-1], out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.proj(x)


class GeomEncoder(nn.Module):

    def __init__(self, d_geom: int = 2, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_geom, 32), nn.Sigmoid(),
            nn.Linear(32, 32), nn.Sigmoid(),
            nn.Linear(32, out_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


class Expert(nn.Module):


    def __init__(self, input_dim: int = 224, hidden_dim: int = 128, output_dim: int = 96, dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.ln3 = nn.LayerNorm(output_dim)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        h = self.act(self.ln1(self.input_proj(x)))
        h = self.dropout(h)


        h2 = self.act(self.ln2(self.hidden(h)))
        h2 = self.dropout(h2)
        h = h + h2


        out = self.ln3(self.output_proj(h))
        return out


class ParameterGate(nn.Module):


    def __init__(self, input_dim: int, n_experts: int, temp: float = 1.0):
        super().__init__()
        self.temp = temp
        self.n_experts = n_experts


        self.ln = nn.LayerNorm(input_dim)


        self.gate_proj = nn.Linear(input_dim, n_experts)


        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.gate_proj.weight, std=0.1)

    def forward(self, x):
        x = self.ln(x)
        logits = self.gate_proj(x)
        return F.softmax(logits / self.temp, dim=-1)


class ParameterTower(nn.Module):


    def __init__(self, input_dim: int = 96, hidden_dims: List[int] = [48, 24], dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.tower = nn.Sequential(*layers)

    def forward(self, x):
        return self.tower(x)


class DAConvAttnNetBasedExtractor(nn.Module):


    def __init__(
        self,
        n_vds_tr: int,
        n_vgs_tr: int,
        d_geom: int = 2,
        n_params: int = len(TARGET_PARAM_NAMES),
        n_experts: int = 4,
        emb_T_dim: int = 96,
        emb_gm_dim: int = 96,
        emb_G_dim: int = 32,
        cnn_hidden: List[int] = [24, 48, 96],
        expert_hidden: int = 128,
        expert_out_dim: int = 96,
        tower_hidden: List[int] = [48, 24],
        gate_temp: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_vds_tr = n_vds_tr
        self.n_vgs_tr = n_vgs_tr
        self.n_params = n_params
        self.n_experts = n_experts


        self.transfer_encoder = ResNetEncoder1D(
            in_channels=n_vds_tr,
            hidden_channels=cnn_hidden,
            out_dim=emb_T_dim,
            dropout=dropout,
        )

        self.gm_encoder = ResNetEncoder1D(
            in_channels=n_vds_tr,
            hidden_channels=cnn_hidden,
            out_dim=emb_gm_dim,
            dropout=dropout,
        )

        self.geom_encoder = GeomEncoder(d_geom=d_geom, out_dim=emb_G_dim)

        fusion_dim = emb_T_dim + emb_gm_dim + emb_G_dim


        self.experts = nn.ModuleList([
            Expert(
                input_dim=fusion_dim,
                hidden_dim=expert_hidden,
                output_dim=expert_out_dim,
                dropout=dropout,
            )
            for _ in range(n_experts)
        ])


        gate_input_dim = n_vds_tr * n_vgs_tr
        self.gates = nn.ModuleList([
            ParameterGate(
                input_dim=gate_input_dim,
                n_experts=n_experts,
                temp=gate_temp,
            )
            for _ in range(n_params)
        ])


        self.towers = nn.ModuleList([
            ParameterTower(
                input_dim=expert_out_dim,
                hidden_dims=tower_hidden,
                dropout=dropout,
            )
            for _ in range(n_params)
        ])


        self.config = {
            "n_vds_tr": n_vds_tr, "n_vgs_tr": n_vgs_tr,
            "d_geom": d_geom, "n_params": n_params, "n_experts": n_experts,
            "emb_T_dim": emb_T_dim, "emb_gm_dim": emb_gm_dim, "emb_G_dim": emb_G_dim,
            "fusion_dim": fusion_dim, "cnn_hidden": cnn_hidden,
            "expert_hidden": expert_hidden, "expert_out_dim": expert_out_dim,
            "tower_hidden": tower_hidden, "gate_input_dim": gate_input_dim,
            "gate_temp": gate_temp, "dropout": dropout,
            "type": "da_convattnnet_based",
        }

    def forward(self, x_T: torch.Tensor, x_gm: torch.Tensor, x_geom: torch.Tensor) -> torch.Tensor:

        emb_T = self.transfer_encoder(x_T)
        emb_gm = self.gm_encoder(x_gm)
        emb_G = self.geom_encoder(x_geom)
        h_fused = torch.cat([emb_T, emb_gm, emb_G], dim=1)


        expert_outs = torch.stack([exp(h_fused) for exp in self.experts], dim=1)


        gate_input = x_T.flatten(1)


        outputs = []
        for k in range(self.n_params):
            w_k = self.gates[k](gate_input)
            f_k = (w_k.unsqueeze(-1) * expert_outs).sum(dim=1)
            p_k = self.towers[k](f_k)
            outputs.append(p_k)

        return torch.cat(outputs, dim=-1)

    def get_all_gate_weights(self, x_T: torch.Tensor) -> torch.Tensor:
        gate_input = x_T.flatten(1)
        return torch.stack([self.gates[k](gate_input) for k in range(self.n_params)], dim=1)

    def print_structure(self):
        print("\n" + "=" * 80)
        print("DA-ConvAttnNet-Based Parameter Extraction Network")
        print("=" * 80)

        print(f"\n[Encoders]")
        print(f"  Transfer current: (B, {self.n_vds_tr}, {self.n_vgs_tr}) → CNN → (B, {self.config['emb_T_dim']})")
        print(f"  Transconductance: (B, {self.n_vds_tr}, {self.n_vgs_tr}) → CNN → (B, {self.config['emb_gm_dim']})")
        print(f"  Geometry:         (B, 2) → MLP → (B, {self.config['emb_G_dim']})")
        print(f"  Fused embedding dimension: {self.config['fusion_dim']}")

        print(f"\n[Experts] Count: {self.n_experts}")
        print(f"  Architecture: {self.config['fusion_dim']}→{self.config['expert_hidden']}→{self.config['expert_hidden']}→{self.config['expert_out_dim']}")
        print(f"  Components: LayerNorm + residual connection + dropout")

        print(f"\n[Parameter Gates] Count: {self.n_params}")
        print(f"  Architecture: LayerNorm → Linear({self.config['gate_input_dim']}→{self.n_experts}) → Softmax")
        print(f"  Weight initialization: standard deviation = 0.1")

        print(f"\n[Regression Heads] Count: {self.n_params}")
        print(f"  Architecture: {self.config['expert_out_dim']}→{self.config['tower_hidden']}→1")
        print(f"  Components: LayerNorm + dropout")


        total = sum(p.numel() for p in self.parameters())
        enc_params = sum(p.numel() for p in self.transfer_encoder.parameters())
        enc_params += sum(p.numel() for p in self.gm_encoder.parameters())
        geom_params = sum(p.numel() for p in self.geom_encoder.parameters())
        expert_params = sum(p.numel() for p in self.experts.parameters())
        gate_params = sum(p.numel() for p in self.gates.parameters())
        tower_params = sum(p.numel() for p in self.towers.parameters())

        print(f"\n[Trainable Parameter Statistics]")
        print(f"  Electrical encoders: {enc_params:,} ({100*enc_params/total:.1f}%)")
        print(f"  Geometry encoder:    {geom_params:,} ({100*geom_params/total:.1f}%)")
        print(f"  Experts:             {expert_params:,} ({100*expert_params/total:.1f}%)")
        print(f"  Parameter gates:     {gate_params:,} ({100*gate_params/total:.1f}%)")
        print(f"  Regression heads:    {tower_params:,} ({100*tower_params/total:.1f}%)")
        print(f"  ─────────────────────")
        print(f"  Total:               {total:,}")
        print(f"  Sample-to-parameter ratio (300K samples): {300000/total:.2f}")

        print("=" * 80)


def compute_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_pred, y_true)


def build_dataloader(X_T, X_gm, X_geom, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X_T).float(),
        torch.from_numpy(X_gm).float(),
        torch.from_numpy(X_geom).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: str = "cuda",
    epochs: int = 300,
    base_lr: float = 0.00001,
    max_lr: float = 0.01,
    grad_clip: float = 1.0,
    patience: int = 20,
    min_improve: float = 0.0,
    accum_steps: int = 8,
    ckpt_path: str = "best_da_convattnnet_based.pth",
    iv_param_names: List[str] = None,
) -> List[Tuple[float, float]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader) // accum_steps

    scheduler = CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=5 * steps_per_epoch,
        mode='triangular2',
        cycle_momentum=False,
    )

    history = []
    best_val = float("inf")
    patience_counter = 0

    effective_batch_size = train_loader.batch_size * accum_steps

    print(f"\n{'=' * 80}")
    print("Training Configuration (DA-ConvAttnNet-Based Parameter Extraction Network)")
    print(f"{'=' * 80}")
    print(f"  Effective batch size:     {effective_batch_size}")
    print(f"  Optimizer:               AdamW (weight_decay=1e-4)")
    print(f"  Learning-rate scheduler: CyclicLR (triangular2)")
    print(f"  Minimum learning rate:   {base_lr}")
    print(f"  Maximum learning rate:   {max_lr}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Min improvement:         {min_improve:.3e}")
    print(f"  Max epochs:              {epochs}")
    print(f"  Device:                  {device}")
    print(f"{'=' * 80}\n")

    def evaluate(loader):
        model.eval()
        total_loss, n_samples = 0.0, 0
        mse_per_param = np.zeros(model.n_params, dtype=np.float64)

        with torch.no_grad():
            for x_T, x_gm, x_geom, y_true in loader:
                x_T, x_gm, x_geom, y_true = [t.to(device) for t in [x_T, x_gm, x_geom, y_true]]
                y_pred = model(x_T, x_gm, x_geom)
                loss = compute_loss(y_pred, y_true)

                bs = x_T.size(0)
                total_loss += loss.item() * bs
                n_samples += bs
                mse_per_param += ((y_pred - y_true).pow(2).cpu().numpy()).sum(axis=0)

        return {
            "loss": total_loss / n_samples,
            "rmse_per_param": np.sqrt(mse_per_param / n_samples),
        }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_train = 0.0, 0
        optimizer.zero_grad()

        for step, (x_T, x_gm, x_geom, y_true) in enumerate(train_loader):
            x_T, x_gm, x_geom, y_true = [t.to(device) for t in [x_T, x_gm, x_geom, y_true]]

            y_pred = model(x_T, x_gm, x_geom)
            loss = compute_loss(y_pred, y_true)

            (loss / accum_steps).backward()

            train_loss += loss.item() * x_T.size(0)
            n_train += x_T.size(0)

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss /= n_train
        val_metrics = evaluate(val_loader)
        val_loss = val_metrics["loss"]
        history.append((train_loss, val_loss))

        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < (best_val - min_improve):
            best_val = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            }, ckpt_path)
        else:
            patience_counter += 1


        print(f"[{epoch:03d}/{epochs}] "
              f"training_mse={train_loss:.6e} validation_mse={val_loss:.6e} best_validation_mse={best_val:.6e} | "
              f"learning_rate={current_lr:.6e} | early_stopping={patience_counter}/{patience}")

        if epoch % 5 == 0 and iv_param_names:
            rmse_str = ", ".join(f"{n}={v:.4f}" for n, v in zip(iv_param_names, val_metrics["rmse_per_param"]))
            print(f"         RMSE: {rmse_str}")

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}")
            break

    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"\n[INFO] Restored the best checkpoint (epoch {ckpt['epoch']}, validation_mse={ckpt['best_val']:.6e})")

    return history


def parse_args():
    ap = argparse.ArgumentParser(description="Train the DA-ConvAttnNet-Based Parameter Extraction Network")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--base-lr", type=float, default=0.00001)
    ap.add_argument("--max-lr", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-improve", type=float, default=2e-5,
                    help="Minimum validation-loss reduction required to update the best checkpoint")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--gate-temp", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", type=str, default="da_convattnnet_based_model.pth")
    ap.add_argument("--ckpt", type=str, default="best_da_convattnnet_based.pth")
    return ap.parse_args()


def run_once(args):
    data = load_normalized_data(Path(args.data))

    print("\nConstructing data loaders...")
    train_loader = build_dataloader(
        data["X_T_train"], data["X_gm_train"], data["X_geom_train"], data["y_train"],
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = build_dataloader(
        data["X_T_val"], data["X_gm_val"], data["X_geom_val"], data["y_val"],
        batch_size=args.batch_size, shuffle=False
    )

    model = DAConvAttnNetBasedExtractor(
        n_vds_tr=data["n_vds_tr"],
        n_vgs_tr=data["n_vgs_tr"],
        d_geom=data["X_geom_train"].shape[1],
        n_params=len(data["iv_param_names"]),
        n_experts=args.n_experts,
        gate_temp=args.gate_temp,
        dropout=args.dropout,
    )
    model.print_structure()

    history = train(
        model, train_loader, val_loader,
        device=args.device,
        epochs=args.epochs,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        grad_clip=args.grad_clip,
        patience=args.patience,
        min_improve=args.min_improve,
        accum_steps=args.accum_steps,
        ckpt_path=args.ckpt,
        iv_param_names=data["iv_param_names"],
    )

    save_obj = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "scalers": data["scalers"],
        "param_names": {"geom": data["geom_param_names"], "iv": data["iv_param_names"]},
        "training": {
            "epochs": len(history),
            "best_val_loss": min(h[1] for h in history) if history else None,
            "args": vars(args),
        },
        "bias_grids": {"vds_tr": data["vds_tr"], "vgs_tr": data["vgs_tr"]},
    }

    torch.save(save_obj, args.save)
    print(f"\n[OK] Model saved to {args.save}")
    print(f"     Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")


def main():
    args = parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
