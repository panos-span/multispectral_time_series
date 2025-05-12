import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchmetrics import F1Score, Accuracy, Precision, Recall
import math
from tqdm import tqdm
from pathlib import Path
import time

# Import from our data_loader.py
from data_loader import (
    preprocess_dataset,
    create_k_fold_loaders,
    set_random_seeds,
    RANDOM_SEED,
    NUM_PIXELS_SAMPLE,
    BATCH_SIZE,
)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
set_random_seeds(RANDOM_SEED)


# Model Architecture Components
class PixelSetMLP(nn.Module):
    """Custom MLP that correctly handles batch normalization for the pixel-set encoder."""

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.dims = [input_dim] + hidden_dims

        # Create layers
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if i < len(self.dims) - 2:  # No BatchNorm and ReLU after last layer
                self.layers.append(nn.BatchNorm1d(self.dims[i + 1]))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B*T*S, C] where:
               B = batch size, T = time steps, S = pixels, C = channels
        """
        # Process through layers with special handling for BatchNorm
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x)  # BatchNorm expects [N, C]
            else:
                x = layer(x)
        return x


class PixelSetEncoder(nn.Module):
    """
    Pixel-Set Encoder for spatial feature extraction from unordered sets of pixels.
    """

    def __init__(
        self,
        input_channels=10,
        mlp1_dims=[32, 64],
        mlp2_dims=[128, 128],
        pooling="mean_std",
        with_extra=False,
        extra_size=4,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.pooling = pooling
        self.with_extra = with_extra
        self.extra_size = extra_size

        # MLP1 for pixel-wise processing
        self.mlp1 = PixelSetMLP(input_channels, mlp1_dims)

        # Calculate pooling dimension
        pool_dim = mlp1_dims[-1] * len(pooling.split("_"))
        if with_extra:
            pool_dim += extra_size

        # MLP2 for processing pooled features
        self.mlp2 = PixelSetMLP(pool_dim, mlp2_dims)

    def forward(self, x, extra=None):
        """
        Args:
            x: Tensor of shape [B, T, C, S] - batch, time, channels, pixels
            extra: Tensor of shape [B, E] - batch, extra features (optional)

        Returns:
            Tensor of shape [B, T, F] - batch, time, output features
        """
        B, T, C, S = x.shape

        # Reshape to process all pixels individually
        x = x.permute(0, 1, 3, 2).reshape(B * T * S, C)  # [B*T*S, C]

        # Process each pixel with MLP1
        x = self.mlp1(x)  # [B*T*S, F]

        # Reshape back to organize by batch, time, pixels
        x = x.view(B * T, S, -1)  # [B*T, S, F]

        # Pool across pixel dimension
        pooled = []
        if "mean" in self.pooling:
            pooled.append(x.mean(dim=1))  # [B*T, F]
        if "std" in self.pooling:
            pooled.append(x.std(dim=1))  # [B*T, F]

        x = torch.cat(pooled, dim=1)  # [B*T, F*num_pooling]

        # Add extra features if available
        if self.with_extra and extra is not None:
            # Repeat extra features for each time step
            extra = extra.unsqueeze(1).repeat(1, T, 1).reshape(B * T, -1)
            x = torch.cat([x, extra], dim=1)

        # Process pooled features
        x = self.mlp2(x)  # [B*T, F_out]

        # Reshape back to [B, T, F_out]
        x = x.view(B, T, -1)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_position=366, T=1000.0):
        super().__init__()
        # Create position encoding table
        pe = torch.zeros(max_position, d_model)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(T) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions):
        """
        positions: [B, T] - batch, time positions (days of year)
        Returns: [B, T, D] - batch, time, features
        """
        return self.pe[positions]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_model = d_model

        # Key, query, value projections
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_q = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, d_model)

        # Master query projection
        self.w_master = nn.Linear(d_k, d_k)

        # Output projection
        self.fc = nn.Linear(n_head * d_k, d_model)

        # Temperature scaling
        self.scale = d_k**-0.5

    def forward(self, x):
        """
        x: [B, T, D] - batch, time, features
        """
        B, T, D = x.shape

        # Project to keys, queries, values
        k = (
            self.w_k(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        )  # [B, H, T, d_k]
        q = (
            self.w_q(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        )  # [B, H, T, d_k]
        v = self.w_v(x).unsqueeze(1).repeat(1, self.n_head, 1, 1)  # [B, H, T, D]

        # Create master query (to get sequence-level representation)
        q_mean = q.mean(dim=2)  # [B, H, d_k]
        q_master = self.w_master(q_mean).unsqueeze(2)  # [B, H, 1, d_k]

        # Calculate attention scores
        attn = torch.matmul(q_master, k.transpose(2, 3)) * self.scale  # [B, H, 1, T]
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn, v)  # [B, H, 1, D]
        output = (
            output.squeeze(2).transpose(1, 2).contiguous().view(B, self.n_head * D)
        )  # [B, H*D]

        # Final projection
        output = self.fc(output)  # [B, D]

        return output, attn.squeeze(2)  # [B, D], [B, H, T]


class TemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        d_model=128,
        n_head=4,
        d_k=32,
        mlp_dims=[128, 128],
        dropout=0.1,
        max_position=366,
        T=1000.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection if necessary
        self.input_proj = (
            nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_position, T)

        # Self-attention module
        self.attention = MultiHeadAttention(d_model, n_head, d_k)

        # Final MLP
        layers = []
        dims = [d_model] + mlp_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(mlp_dims[-1])

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        """
        x: [B, T, D_in] - batch, time, input features
        positions: [B, T] - batch, time positions (days of year)
        """
        # Project input if necessary
        x = self.input_proj(x)  # [B, T, D]

        # Add positional encoding
        pos = self.pos_encoder(positions)  # [B, T, D]
        x = x + pos

        # Apply normalization and dropout
        x = self.norm1(x)
        x = self.dropout(x)

        # Apply attention
        attn_output, attention = self.attention(x)  # [B, D], [B, H, T]

        # Apply MLP
        output = self.mlp(attn_output)  # [B, D_out]

        # Final normalization
        output = self.norm2(output)

        return output, attention


class PSETAEModel(nn.Module):
    """Complete model combining PSE and TAE for satellite time series classification."""

    def __init__(
        self,
        num_classes,
        input_channels=10,
        pse_mlp1=[32, 64],
        pse_mlp2=[128, 128],
        tae_d_model=128,
        tae_n_head=4,
        tae_d_k=32,
        tae_mlp=[128, 128],
        cls_mlp=[128, 64, 32],
        with_extra=False,
        extra_size=4,
        dropout=0.1,
        max_position=366,
    ):
        super().__init__()

        # Pixel-Set Encoder
        self.pse = PixelSetEncoder(
            input_channels=input_channels,
            mlp1_dims=pse_mlp1,
            mlp2_dims=pse_mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )

        # Temporal Attention Encoder
        self.tae = TemporalAttentionEncoder(
            in_channels=pse_mlp2[-1],
            d_model=tae_d_model,
            n_head=tae_n_head,
            d_k=tae_d_k,
            mlp_dims=tae_mlp,
            dropout=dropout,
            max_position=max_position,
        )

        # Classification MLP
        layers = []
        dims = [tae_mlp[-1]] + cls_mlp + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.classifier = nn.Sequential(*layers)

    def forward(self, pixels, positions, extra=None):
        """
        pixels: [B, T, C, S] - batch, time, channels, pixels
        positions: [B, T] - batch, time positions (days of year)
        extra: [B, E] - batch, extra features (optional)
        """
        # Extract spatial features
        x = self.pse(pixels, extra)  # [B, T, F]

        # Extract temporal features
        x, attention = self.tae(x, positions)  # [B, F], [B, H, T]

        # Classification
        logits = self.classifier(x)  # [B, num_classes]

        return logits, attention


# Training and Evaluation Functions
def train_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # Get data
        pixels = batch["pixels"].to(device)
        positions = batch["positions"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        logits, _ = model(pixels, positions)
        loss = criterion(logits, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        epoch_loss += loss.item() * labels.size(0)

        # Update progress bar
        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

    return epoch_loss / total, correct / total


def evaluate(model, loader, criterion, device, metrics=None):
    """Evaluate model on validation or test set."""
    model.eval()
    epoch_loss = 0
    total = 0
    all_preds = []
    all_labels = []
    all_attention = []

    if metrics:
        for metric in metrics.values():
            metric.reset()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Get data
            pixels = batch["pixels"].to(device)
            positions = batch["positions"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            logits, attention = model(pixels, positions)
            loss = criterion(logits, labels)

            # Statistics
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            epoch_loss += loss.item() * labels.size(0)

            # Store predictions and labels
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_attention.append(attention.cpu())

            # Update metrics if provided
            if metrics:
                for metric in metrics.values():
                    metric.update(preds, labels)

    # Concatenate results
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute metrics
    results = {"loss": epoch_loss / total}
    if metrics:
        for name, metric in metrics.items():
            results[name] = metric.compute().item()

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return results, cm, (all_preds, all_labels, all_attention)


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    metrics,
    idx_to_crop,
    num_epochs=15,
    lr_scheduler=None,
    early_stopping_patience=5,
    model_save_path="best_model.pt",
):
    """Train and evaluate model with early stopping."""
    train_losses = []
    train_accs = []
    val_metrics_history = []
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train for one epoch
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_time = time.time() - start_time

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(
            f"Training loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, time: {train_time:.2f}s"
        )

        # Validate
        start_time = time.time()
        val_metrics, confusion_mat, _ = evaluate(
            model, val_loader, criterion, device, metrics
        )
        val_time = time.time() - start_time

        val_metrics_history.append(val_metrics)

        print("Validation metrics:")
        for name, value in val_metrics.items():
            print(f"  {name}: {value:.4f}")
        print(f"Validation time: {val_time:.2f}s")

        # Learning rate scheduling
        if lr_scheduler is not None:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Current learning rate: {current_lr:.6f}")

        # Check for improvement
        current_f1 = val_metrics.get("f1_weighted", val_metrics.get("f1_micro", 0))
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                model_save_path,
            )

            print(f"Best model saved at epoch {epoch+1} with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{early_stopping_patience}"
            )

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
                break

    # Load best model before returning
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_metrics_history": val_metrics_history,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
    }


def run_k_fold_cross_validation(
    fold_loaders, num_classes, idx_to_crop, num_epochs=15, device="cuda"
):
    """Run k-fold cross-validation and return aggregated results."""
    k = len(fold_loaders)
    all_fold_results = []
    all_fold_metrics = []
    all_confusion_matrices = []

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{k}")
        print(f"{'='*50}")

        # Initialize metrics
        metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
            "f1_micro": F1Score(
                average="micro", task="multiclass", num_classes=num_classes
            ).to(device),
            "f1_weighted": F1Score(
                average="weighted", task="multiclass", num_classes=num_classes
            ).to(device),
            "precision_micro": Precision(
                average="micro", task="multiclass", num_classes=num_classes
            ).to(device),
            "precision_weighted": Precision(
                average="weighted", task="multiclass", num_classes=num_classes
            ).to(device),
            "recall_micro": Recall(
                average="micro", task="multiclass", num_classes=num_classes
            ).to(device),
            "recall_weighted": Recall(
                average="weighted", task="multiclass", num_classes=num_classes
            ).to(device),
        }

        # Model parameters - FIXED from previous implementation
        model_params = {
            "num_classes": num_classes,
            "input_channels": 10,  # Sentinel-2 bands
            "pse_mlp1": [32, 64],  # FIXED: removed the input_channels from mlp1_dims
            "pse_mlp2": [128, 128],
            "tae_d_model": 128,
            "tae_n_head": 4,
            "tae_d_k": 32,
            "tae_mlp": [128, 128],
            "cls_mlp": [128, 64, 32],
            "with_extra": False,
            "dropout": 0.2,
            "max_position": 366,  # Days in a year
        }

        # Initialize model
        model = PSETAEModel(**model_params).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        # Train and evaluate
        model_save_path = f"best_model_fold{fold+1}.pt"
        fold_result = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            metrics=metrics,
            idx_to_crop=idx_to_crop,
            num_epochs=num_epochs,
            lr_scheduler=scheduler,
            early_stopping_patience=5,
            model_save_path=model_save_path,
        )

        # Final evaluation on validation set
        final_metrics, confusion_mat, _ = evaluate(
            model, val_loader, criterion, device, metrics
        )
        all_fold_metrics.append(final_metrics)
        all_confusion_matrices.append(confusion_mat)
        all_fold_results.append(fold_result)

        print(f"\nFold {fold+1} Final Metrics:")
        for name, value in final_metrics.items():
            print(f"  {name}: {value:.4f}")

    # Calculate mean and std of metrics across folds
    all_metric_names = all_fold_metrics[0].keys()
    mean_metrics = {
        name: np.mean([fold_metric[name] for fold_metric in all_fold_metrics])
        for name in all_metric_names
    }
    std_metrics = {
        name: np.std([fold_metric[name] for fold_metric in all_fold_metrics])
        for name in all_metric_names
    }

    # Calculate average confusion matrix
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)

    print("\nOverall Cross-Validation Results:")
    for name in sorted(mean_metrics.keys()):
        print(f"{name}: {mean_metrics[name]:.4f} ± {std_metrics[name]:.4f}")

    return {
        "fold_results": all_fold_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "avg_confusion_matrix": avg_confusion_matrix,
    }


# Main execution
def main():
    # 1. Preprocess dataset
    (
        kept_parcel_ids,
        filtered_labels,
        crop_to_idx,
        idx_to_crop,
        num_classes,
        dates_doy,
    ) = preprocess_dataset()

    # 2. Create k-fold data loaders
    fold_loaders = create_k_fold_loaders(
        parcel_ids=kept_parcel_ids,
        labels_dict=filtered_labels,
        crop_to_idx=crop_to_idx,
        data_path=Path("timematch_data") / "denmark" / "32VNH" / "2017" / "data",
        dates_doy=dates_doy,
        num_folds=5,
        batch_size=BATCH_SIZE,
        num_pixels_sample=NUM_PIXELS_SAMPLE,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 2,
        random_seed=RANDOM_SEED,
        max_pixel_value=10000.0,  # Scaling value for normalization
    )

    # 3. Run cross-validation
    results = run_k_fold_cross_validation(
        fold_loaders=fold_loaders,
        num_classes=num_classes,
        idx_to_crop=idx_to_crop,
        num_epochs=15,
        device=device,
    )

    # 4. Print final results
    print("\nFinal Results Summary:")
    print(f"Number of classes: {num_classes}")
    print(f"Number of folds: {len(fold_loaders)}")

    for metric, value in sorted(results["mean_metrics"].items()):
        print(f"{metric}: {value:.4f} ± {results['std_metrics'][metric]:.4f}")


if __name__ == "__main__":
    main()
