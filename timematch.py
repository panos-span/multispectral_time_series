# timematch.py - Updated with Fast Loading Support
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall
from tqdm import tqdm

# Updated imports for enhanced normalization and fast loading
from data_loader import (
    BATCH_SIZE,
    DATA_PATH,
    INPUT_CHANNELS,
    NUM_FOLDS,
    NUM_PIXELS_SAMPLE,
    NUM_WORKERS,
    RANDOM_SEED,
    NORMALIZATION_STRATEGY,
    create_enhanced_k_fold_loaders,
    preprocess_dataset_with_normalization,
    load_preprocessed_data,          # New import
    check_preprocessed_data_exists,  # New import
    set_random_seeds,
)

# --- Configuration ---
# Fast loading option - set to True to load preprocessed data directly
USE_FAST_LOADING = True  # Set to False to always preprocess from scratch

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Set random seeds for reproducibility
set_random_seeds(RANDOM_SEED)

# PSE Configuration
PSE_MLP1_DIMS = [32, 64]
PSE_MLP2_DIMS = [256, 128]         
PSE_OUTPUT_DIM = PSE_MLP2_DIMS[-1]

# LTAE Configuration
LTAE_D_MODEL = PSE_OUTPUT_DIM
LTAE_NHEAD = 8
LTAE_D_K = 16
LTAE_MLP_DIMS = [512, 256, 128]
LTAE_DROPOUT = 0.2

# Classifier Head Configuration
CLASSIFIER_HIDDEN_DIMS = [64, 32]
CLASSIFIER_DROPOUT = 0.1

# Training Configuration
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5


# --- Model Components (unchanged from previous version) ---
class MLP(nn.Module):
    """Simple MLP module with batch normalization and activation."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        use_batchnorm=True,
        activation=nn.ReLU,
        dropout=0.0,
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PixelSetEncoder(nn.Module):
    """Encodes spatial/spectral info per time step."""

    def __init__(self, input_dim, mlp1_dims, mlp2_dims, pooling="mean_std"):
        super().__init__()
        self.input_dim = input_dim
        self.pooling = pooling

        # First MLP to process individual pixels
        self.mlp1 = MLP(
            input_dim=input_dim,
            hidden_dims=mlp1_dims[:-1],
            output_dim=mlp1_dims[-1],
            use_batchnorm=True,
        )

        # Calculate pooling output dimension
        if pooling == "mean_std":
            pooling_dim = mlp1_dims[-1] * 2
        elif pooling in ["mean", "max"]:
            pooling_dim = mlp1_dims[-1]
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        # Second MLP to process pooled features
        self.mlp2 = MLP(
            input_dim=pooling_dim,
            hidden_dims=mlp2_dims[:-1],
            output_dim=mlp2_dims[-1],
            use_batchnorm=True,
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, C, S] - batch, time, channels, pixels
        Returns:
            Tensor of shape [B, T, F] - batch, time, output features
        """
        B, T, C, S = x.shape

        # Reshape to process all pixels together
        x = x.permute(0, 1, 3, 2).contiguous().view(B * T * S, C)

        # Process each pixel with mlp1
        x = self.mlp1(x)

        # Reshape back to organize by batch, time, pixels
        F = x.shape[-1]
        x = x.view(B * T, S, F)

        # Apply pooling across pixel dimension
        if self.pooling == "mean_std":
            mean = x.mean(dim=1)
            std = x.std(dim=1)
            x_pooled = torch.cat([mean, std], dim=1)
        elif self.pooling == "mean":
            x_pooled = x.mean(dim=1)
        elif self.pooling == "max":
            x_pooled = x.max(dim=1)[0]

        # Process pooled features with mlp2
        x = self.mlp2(x_pooled)

        # Reshape to [B, T, F_out]
        x = x.view(B, T, -1)

        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding based on day-of-year."""

    def __init__(self, d_model, max_len=366 + 1, T=1000.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(T) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions):
        """
        Args:
            positions: [B, T] tensor with day-of-year values
        Returns:
            [B, T, D] positional encodings
        """
        pos_indices = positions.clamp(0, self.pe.shape[0] - 1).long()
        return self.pe[pos_indices]


class LightweightTemporalAttentionEncoder(nn.Module):
    """Implementation of Lightweight Temporal Attention Encoder (LTAE)."""

    def __init__(
        self,
        in_channels,
        d_model=None,
        n_head=8,
        d_k=16,
        mlp_dims=[256, 128],
        dropout=0.2,
        max_position=366,
        T=1000.0,
    ):
        super().__init__()
        self.in_channels = in_channels

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Linear(in_channels, d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=self.d_model, max_len=max_position, T=T
        )

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )

        layers = []
        dims = [n_head * self.d_model] + mlp_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.inlayernorm = nn.LayerNorm(self.d_model)
        self.outlayernorm = nn.LayerNorm(mlp_dims[-1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        """
        Args:
            x: [B, T, C] input features
            positions: [B, T] day-of-year positions
        Returns:
            [B, F] output features, [B, H, T] attention weights
        """
        x = self.inlayernorm(x)
        pos_encoding = self.positional_encoding(positions)
        enc_output = x + pos_encoding

        if self.inconv is not None:
            enc_output = self.inconv(enc_output)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)
        enc_output = enc_output.permute(1, 0, 2).contiguous().view(x.shape[0], -1)
        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        return enc_output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention with a master query mechanism."""

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / d_k))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(n_head * d_k), nn.Linear(n_head * d_k, n_head * d_k)
        )

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = self.fc1_q(q).view(sz_b, seq_len, n_head, d_k)
        q = q.mean(dim=1).squeeze()
        q = self.fc2(q.view(sz_b, n_head * d_k)).view(sz_b, n_head, d_k)
        q = q.permute(1, 0, 2).contiguous().view(n_head * sz_b, d_k)

        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)

        v = v.repeat(n_head, 1, 1)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, 1, d_in)
        output = output.squeeze(dim=2)

        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class TimeMatchModel(nn.Module):
    """Complete model combining PSE and LTAE for satellite time series classification."""

    def __init__(
        self,
        num_classes,
        input_channels=INPUT_CHANNELS,
        pse_mlp1_dims=PSE_MLP1_DIMS,
        pse_mlp2_dims=PSE_MLP2_DIMS,
        ltae_d_model=LTAE_D_MODEL,
        ltae_nhead=LTAE_NHEAD,
        ltae_d_k=LTAE_D_K,
        ltae_mlp_dims=LTAE_MLP_DIMS,
        ltae_dropout=LTAE_DROPOUT,
        classifier_hidden_dims=CLASSIFIER_HIDDEN_DIMS,
        classifier_dropout=CLASSIFIER_DROPOUT,
        max_position=366,
    ):
        super().__init__()

        self.pse = PixelSetEncoder(
            input_dim=input_channels, mlp1_dims=pse_mlp1_dims, mlp2_dims=pse_mlp2_dims
        )

        self.ltae = LightweightTemporalAttentionEncoder(
            in_channels=pse_mlp2_dims[-1],
            d_model=ltae_d_model,
            n_head=ltae_nhead,
            d_k=ltae_d_k,
            mlp_dims=ltae_mlp_dims,
            dropout=ltae_dropout,
            max_position=max_position,
        )

        self.classifier = MLP(
            input_dim=ltae_mlp_dims[-1],
            hidden_dims=classifier_hidden_dims,
            output_dim=num_classes,
            use_batchnorm=False,
            dropout=classifier_dropout,
        )

    def forward(self, pixels, positions):
        """
        Args:
            pixels: [B, T, C, S] tensor - batch, time, channels, pixels
            positions: [B, T] tensor - batch, time positions (days of year)
        Returns:
            logits: [B, num_classes] - classification logits
            attention: attention weights from LTAE
        """
        x = self.pse(pixels)
        x, attention = self.ltae(x, positions)
        logits = self.classifier(x)
        return logits, attention


# --- Helper Functions (unchanged) ---
def initialize_metrics(num_classes, device, prefix="val"):
    """Initialize metrics dictionary for evaluation."""
    metrics = {
        f"{prefix}_acc": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        f"{prefix}_prec_micro": Precision(task="multiclass", num_classes=num_classes, average="micro").to(device),
        f"{prefix}_recall_micro": Recall(task="multiclass", num_classes=num_classes, average="micro").to(device),
        f"{prefix}_f1_micro": F1Score(task="multiclass", num_classes=num_classes, average="micro").to(device),
        f"{prefix}_prec_weighted": Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device),
        f"{prefix}_recall_weighted": Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device),
        f"{prefix}_f1_weighted": F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device),
    }
    return metrics


def display_confusion_matrix(conf_matrix, class_names, fold_num):
    """Display confusion matrix as a heatmap."""
    if conf_matrix is None:
        print(f"Warning: No confusion matrix available for fold {fold_num}")
        return

    n_classes = conf_matrix.shape[0]

    if len(class_names) > n_classes:
        class_names = class_names[:n_classes]
    elif len(class_names) < n_classes:
        class_names = list(class_names) + [f"unknown_{i}" for i in range(len(class_names), n_classes)]

    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10))
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="viridis")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=10)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=10)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title(f"Confusion Matrix - Fold {fold_num}")
        plt.tight_layout()
        os.makedirs("./plots", exist_ok=True)
        plot_path = Path("./plots") / f"confusion_matrix_fold{fold_num}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved confusion matrix for fold {fold_num} to confusion_matrix_fold{fold_num}.png")
    except Exception as e:
        print(f"Error generating heatmap for fold {fold_num}: {e}")


# --- Training and Validation Functions (unchanged) ---
def train_epoch(model, loader, criterion, optimizer, device, metrics=None):
    """Train model for one epoch."""
    model.train()
    epoch_loss = 0.0
    samples_processed = 0

    if metrics:
        for metric in metrics.values():
            metric.reset()

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        pixels = batch["pixels"].to(device)
        positions = batch["positions"].to(device)
        labels = batch["label"].to(device)
        batch_size = labels.size(0)

        if batch_size == 0:
            continue

        logits, _ = model(pixels, positions)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        if metrics:
            for metric in metrics.values():
                metric.update(preds, labels)

        epoch_loss += loss.item() * batch_size
        samples_processed += batch_size

        if metrics and "train_acc" in metrics:
            try:
                acc_val = metrics["train_acc"].compute()
                if acc_val.numel() == 1:
                    current_acc = acc_val.item()
                else:
                    current_acc = acc_val.mean().item()
                pbar.set_postfix({"loss": loss.item(), "acc": f"{current_acc:.4f}"})
            except Exception as e:
                pbar.set_postfix({"loss": loss.item()})
        else:
            pbar.set_postfix({"loss": loss.item()})

    avg_loss = epoch_loss / samples_processed if samples_processed > 0 else 0

    metrics_results = {}
    if metrics:
        for name, metric in metrics.items():
            if not isinstance(metric, ConfusionMatrix):
                try:
                    metric_val = metric.compute()
                    if metric_val.numel() == 1:
                        metrics_results[name] = metric_val.item()
                    else:
                        metrics_results[name] = metric_val.mean().item()
                except Exception as e:
                    print(f"Warning: Could not compute metric {name}: {e}")

    return avg_loss, metrics_results


def validate(model, loader, criterion, device, metrics=None):
    """Evaluate model on validation data."""
    model.eval()
    val_loss = 0.0
    samples_processed = 0
    all_attention = []
    all_preds = []
    all_labels = []

    if metrics:
        for metric in metrics.values():
            metric.reset()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            pixels = batch["pixels"].to(device)
            positions = batch["positions"].to(device)
            labels = batch["label"].to(device)
            batch_size = labels.size(0)

            if batch_size == 0:
                continue

            logits, attention = model(pixels, positions)
            loss = criterion(logits, labels)

            all_attention.append(attention.cpu())
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            if metrics:
                for name, metric in metrics.items():
                    if not isinstance(metric, ConfusionMatrix):
                        try:
                            metric.update(preds, labels)
                        except Exception as e:
                            print(f"Warning: Could not update metric {name}: {e}")

            val_loss += loss.item() * batch_size
            samples_processed += batch_size

    avg_loss = val_loss / samples_processed if samples_processed > 0 else 0

    metrics_results = {}
    confusion_matrix = None

    if metrics:
        for name, metric in metrics.items():
            if not isinstance(metric, ConfusionMatrix):
                try:
                    metric_val = metric.compute()
                    if metric_val.numel() == 1:
                        metrics_results[name] = metric_val.item()
                    else:
                        metrics_results[name] = metric_val.mean().item()
                except Exception as e:
                    print(f"Warning: Could not compute metric {name}: {e}")

    if all_preds and all_labels:
        try:
            import sklearn.metrics
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            confusion_matrix = sklearn.metrics.confusion_matrix(
                all_labels, all_preds, labels=list(range(metrics.get("val_acc", {}).num_classes if metrics else 0))
            )
        except Exception as e:
            print(f"Warning: Could not compute confusion matrix: {e}")

    return avg_loss, metrics_results, confusion_matrix, all_attention


def plot_fold_history(history, fold_num, output_dir="plots/history"):
    """Plots training and validation loss over epochs for a fold."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path_loss = Path(output_dir) / f"loss_curve_fold_{fold_num}.png"

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], "bo-", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "ro-", label="Validation Loss")
    plt.title(f"Fold {fold_num}: Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path_loss)
    plt.close()
    print(f"Saved loss curve for fold {fold_num} to {save_path_loss}")


def train_validate_fold(
    fold_num,
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    num_classes,
    idx_to_crop,
    num_epochs,
    early_stopping_patience=5,
):
    """Train and validate model for one fold with early stopping."""
    print(f"\n----- Fold {fold_num + 1}/{NUM_FOLDS} -----")

    train_metrics = initialize_metrics(num_classes, device, prefix="train")
    val_metrics = initialize_metrics(num_classes, device, prefix="val")

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    best_confusion_matrix = None
    history = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        start_time = time.time()
        train_loss, train_metrics_results = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics
        )
        train_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_metrics"].append(train_metrics_results)

        print(f"Training: loss={train_loss:.4f}, time={train_time:.2f}s")
        for name, value in train_metrics_results.items():
            print(f"  {name}={value:.4f}", end="\n")
        print()

        start_time = time.time()
        val_loss, val_metrics_results, confusion_mat, _ = validate(
            model, val_loader, criterion, device, val_metrics
        )
        val_time = time.time() - start_time

        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics_results)

        print(f"Validation: loss={val_loss:.4f}, time={val_time:.2f}s")
        for name, value in val_metrics_results.items():
            print(f"  {name}={value:.4f}", end="\n")
        print()

        if scheduler is not None:
            scheduler.step(val_loss)

        current_val_f1 = val_metrics_results.get("val_f1_weighted", 0.0)
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            best_confusion_matrix = confusion_mat
            patience_counter = 0
            print(f"New best model! F1={best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{early_stopping_patience} epochs")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nFold {fold_num + 1} completed. Best val F1: {best_val_f1:.4f} at epoch {best_epoch+1}")

    if best_confusion_matrix is not None:
        class_names = list(idx_to_crop.values())
        display_confusion_matrix(best_confusion_matrix, class_names, fold_num + 1)
        
    plot_fold_history(history, fold_num + 1)

    best_metrics = history["val_metrics"][best_epoch]

    return best_metrics, history


def run_k_fold_cross_validation(fold_loaders, num_classes, idx_to_crop):
    """Run k-fold cross-validation and aggregate results."""
    all_fold_metrics = []

    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold_idx+1}/{len(fold_loaders)}")
        print(f"{'='*50}")

        model = TimeMatchModel(num_classes=num_classes).to(DEVICE)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {param_count:,} trainable parameters")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=EARLY_STOPPING_PATIENCE // 2,
        )

        fold_metrics, fold_history = train_validate_fold(
            fold_num=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            num_classes=num_classes,
            idx_to_crop=idx_to_crop,
            num_epochs=NUM_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )

        all_fold_metrics.append(fold_metrics)

        os.makedirs("models", exist_ok=True)
        fold_model_path = Path("models") / f"model_fold{fold_idx+1}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "fold_metrics": fold_metrics,
                "fold_history": fold_history,
            },
            fold_model_path,
        )

    metric_names = all_fold_metrics[0].keys()
    mean_metrics = {name: np.mean([fold[name] for fold in all_fold_metrics]) for name in metric_names}
    std_metrics = {name: np.std([fold[name] for fold in all_fold_metrics]) for name in metric_names}

    print("\nOverall Cross-Validation Results:")
    for name in sorted(mean_metrics.keys()):
        print(f"{name}: {mean_metrics[name]:.4f} ± {std_metrics[name]:.4f}")

    return {
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "all_fold_metrics": all_fold_metrics,
    }


# --- Data Loading Strategy ---
def load_or_preprocess_data():
    """
    Smart data loading that tries fast loading first, falls back to preprocessing.
    
    Returns:
        Tuple of preprocessed data components
    """
    if USE_FAST_LOADING and check_preprocessed_data_exists():
        print("\n🚀 Fast Loading Enabled - Loading Preprocessed Data")
        print("=" * 60)
        try:
            return load_preprocessed_data()
        except Exception as e:
            print(f"❌ Fast loading failed: {e}")
            print("   Falling back to preprocessing from scratch...")
    
    print("\n⚙️  Preprocessing Data from Scratch")
    print("=" * 60)
    print("💡 Tip: Run 'python data_loader.py' once to enable fast loading")
    
    return preprocess_dataset_with_normalization()


# --- Main Execution ---
if __name__ == "__main__":
    print(f"\n🎯 TimeMatch Training Pipeline")
    print(f"   Device: {DEVICE}")
    print(f"   Normalization: {NORMALIZATION_STRATEGY}")
    print(f"   Fast Loading: {'✅ Enabled' if USE_FAST_LOADING else '❌ Disabled'}")
    
    # 1. Load or preprocess data
    print("\n--- 1. Data Loading ---")
    (
        kept_parcel_ids,
        filtered_labels,
        crop_to_idx,
        idx_to_crop,
        num_classes,
        dates_doy,
        normalizer,
    ) = load_or_preprocess_data()

    print(f"\n📊 Dataset Summary:")
    print(f"   Classes: {num_classes}")
    print(f"   Parcels: {len(kept_parcel_ids)}")
    print(f"   Normalization: {normalizer.strategy}")
    print(f"   Classes: {list(idx_to_crop.values())}")

    # 2. Create k-fold data loaders
    print("\n--- 2. Creating Data Loaders ---")
    fold_loaders = create_enhanced_k_fold_loaders(
        parcel_ids=kept_parcel_ids,
        labels_dict=filtered_labels,
        crop_to_idx=crop_to_idx,
        data_path=DATA_PATH,
        dates_doy=dates_doy,
        normalizer=normalizer,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,
        num_pixels_sample=NUM_PIXELS_SAMPLE,
        num_workers=NUM_WORKERS,
        random_seed=RANDOM_SEED,
    )

    # 3. Run k-fold cross-validation
    print("\n--- 3. Running K-Fold Cross-Validation ---")
    if fold_loaders:
        results = run_k_fold_cross_validation(
            fold_loaders=fold_loaders, num_classes=num_classes, idx_to_crop=idx_to_crop
        )

        # Print final summary and save it
        print("\n--- Final Results Summary ---")
        print(f"Normalization Strategy: {normalizer.strategy}")
        print(f"Mean Validation Metrics across {NUM_FOLDS} Folds:")
        
        file_path = Path(f"results_summary_{normalizer.strategy}.txt")
        with open(file_path, "w") as f:
            f.write(f"Normalization Strategy: {normalizer.strategy}\n")
            f.write(f"Fast Loading Used: {USE_FAST_LOADING and check_preprocessed_data_exists()}\n")
            f.write(f"Mean Validation Metrics across {NUM_FOLDS} Folds:\n")
            for metric, value in sorted(results["mean_metrics"].items()):
                print(f"  {metric}: {value:.4f} ± {results['std_metrics'][metric]:.4f}")
                f.write(f"  {metric}: {value:.4f} ± {results['std_metrics'][metric]:.4f}\n")
    else:
        print("❌ Error: No data loaders were created.")

    # 4. Save the best fold results
    if results and "all_fold_metrics" in results:
        best_fold_metrics = max(results["all_fold_metrics"], key=lambda x: x.get("val_f1_weighted", 0))
        print(f"\n🏆 Best Fold Metrics:")
        for metric, value in sorted(best_fold_metrics.items()):
            print(f"  {metric}: {value:.4f}")

    print(f"\n✅ Training completed!")
    if USE_FAST_LOADING and check_preprocessed_data_exists():
        print(f"   ⚡ Fast loading was used - saved preprocessing time")
    else:
        print(f"   💡 Run 'python data_loader.py' to enable fast loading for next time")