# model.py

import os
import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any
from collections import Counter
import numpy as np # For K-fold aggregation
import pandas as pd # For displaying confusion matrix nicely
import matplotlib.pyplot as plt # For plotting confusion matrix
import seaborn as sns # For plotting confusion matrix

# Import TorchMetrics
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

# Import necessary components from the data loading script
from data_loader import (
    preprocess_dataset,
    create_k_fold_loaders,
    set_random_seeds,
    INPUT_CHANNELS,
    NUM_TIMESTEPS,
    NUM_FOLDS,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    PIXEL_NORMALIZATION_MAX,
    DATA_PATH,
    NUM_PIXELS_SAMPLE
)

# --- Configuration ---
# PSE Configuration
PSE_MLP1_DIMS = [32, 64]
PSE_MLP2_DIMS = [128]
PSE_OUTPUT_DIM = PSE_MLP2_DIMS[-1] # d_model for Transformer

# Transformer Configuration
TRANSFORMER_NHEAD = 4
TRANSFORMER_DIM_FEEDFORWARD = PSE_OUTPUT_DIM * 4
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1

# Classifier Head Configuration
CLASSIFIER_HIDDEN_DIMS = [64]

# Training Configuration
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15 # Increased slightly for better convergence, adjust as needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Components (MLP, PixelSetEncoder, SinusoidalPositionalEncoding - Keep as before) ---
class MLP(nn.Module):
    """Helper module for Multi-Layer Perceptrons."""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, use_batchnorm: bool = True, activation: nn.Module = nn.ReLU, dropout_rate: float = 0.0):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation:
                layers.append(activation(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class PixelSetEncoder(nn.Module):
    """Encodes spatial/spectral info per time step."""
    def __init__(self, input_dim: int, mlp1_dims: List[int], mlp2_dims: List[int], pooling: str = 'mean_std'):
        super().__init__()
        self.pooling = pooling
        if not mlp1_dims: raise ValueError("mlp1_dims cannot be empty")
        if not mlp2_dims: raise ValueError("mlp2_dims cannot be empty")
        self.mlp1_output_dim = mlp1_dims[-1]
        self.mlp1 = MLP(input_dim, mlp1_dims[:-1], mlp1_dims[-1], use_batchnorm=True)
        if pooling == 'mean_std': mlp2_input_dim = self.mlp1_output_dim * 2
        elif pooling in ['mean', 'max']: mlp2_input_dim = self.mlp1_output_dim
        else: raise ValueError(f"Unsupported pooling type: {pooling}")
        self.mlp2 = MLP(mlp2_input_dim, mlp2_dims[:-1], mlp2_dims[-1], use_batchnorm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, S = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B * T * S, C)
        x = self.mlp1(x)
        x = x.view(B, T, S, self.mlp1_output_dim)
        if self.pooling == 'mean_std':
            mean = x.mean(dim=2)
            std = x.std(dim=2)
            x_pooled = torch.cat((mean, std), dim=-1)
        elif self.pooling == 'mean': x_pooled = x.mean(dim=2)
        elif self.pooling == 'max': x_pooled = x.max(dim=2)[0]
        else: raise ValueError(f"Unsupported pooling type: {self.pooling}")
        mlp2_input_dim = x_pooled.shape[-1]
        x_pooled = x_pooled.view(B * T, mlp2_input_dim)
        output = self.mlp2(x_pooled)
        output = output.view(B, T, -1)
        return output

class SinusoidalPositionalEncoding(nn.Module):
    """Injects positional information based on day-of-year."""
    def __init__(self, d_model: int, max_len: int = 366 + 1): # Max days + buffer
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        pos_indices = positions.clamp(0, self.pe.shape[0] - 1).long()
        return self.pe[pos_indices]

class TimeSeriesTransformer(nn.Module):
    """Combines PSE, Temporal Encoding, Transformer, and Classifier."""
    def __init__(self, num_classes: int,
                 pse_input_dim: int = INPUT_CHANNELS,
                 pse_mlp1_dims: List[int] = PSE_MLP1_DIMS,
                 pse_mlp2_dims: List[int] = PSE_MLP2_DIMS,
                 d_model: int = PSE_OUTPUT_DIM, # Dimension for Transformer
                 nhead: int = TRANSFORMER_NHEAD,
                 num_encoder_layers: int = TRANSFORMER_NUM_LAYERS,
                 dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
                 transformer_dropout: float = TRANSFORMER_DROPOUT,
                 classifier_hidden_dims: List[int] = CLASSIFIER_HIDDEN_DIMS,
                 classifier_dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.pse = PixelSetEncoder(pse_input_dim, pse_mlp1_dims, pse_mlp2_dims, pooling='mean_std')
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=NUM_TIMESTEPS + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=transformer_dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.mlp_head = MLP(
            input_dim=d_model, hidden_dims=classifier_hidden_dims, output_dim=num_classes,
            use_batchnorm=False, dropout_rate=classifier_dropout
        )
        self.num_classes = num_classes # Store num_classes for convenience

    def forward(self, pixels: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B = pixels.shape[0]
        T_actual = pixels.shape[1]
        x = self.pse(pixels) # (B, T_actual, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        cls_pos = torch.zeros((B, 1), dtype=torch.long, device=positions.device)
        actual_pos = positions[:, :T_actual] + 1 # Shift days to 1-based index
        transformer_positions = torch.cat([cls_pos, actual_pos], dim=1)
        pos_encoding = self.positional_encoding(transformer_positions) # (B, T_actual+1, d_model)
        x_with_cls = torch.cat([cls_tokens, x], dim=1) # (B, T_actual+1, d_model)
        transformer_input = x_with_cls + pos_encoding
        transformer_output = self.transformer_encoder(transformer_input)
        cls_output = transformer_output[:, 0, :] # (B, d_model) - Output for CLS token
        logits = self.mlp_head(cls_output) # (B, num_classes)
        return logits

# --- Metrics Initialization ---
def initialize_metrics(num_classes: int, device: torch.device, prefix: str = 'val') -> Dict[str, torchmetrics.Metric]:
    """Initializes a dictionary of metrics."""
    metrics = {
        f'{prefix}_acc': Accuracy(task="multiclass", num_classes=num_classes).to(device),
        f'{prefix}_prec_micro': Precision(task="multiclass", num_classes=num_classes, average='micro').to(device),
        f'{prefix}_recall_micro': Recall(task="multiclass", num_classes=num_classes, average='micro').to(device),
        f'{prefix}_f1_micro': F1Score(task="multiclass", num_classes=num_classes, average='micro').to(device),
        f'{prefix}_prec_weighted': Precision(task="multiclass", num_classes=num_classes, average='weighted').to(device),
        f'{prefix}_recall_weighted': Recall(task="multiclass", num_classes=num_classes, average='weighted').to(device),
        f'{prefix}_f1_weighted': F1Score(task="multiclass", num_classes=num_classes, average='weighted').to(device),
        f'{prefix}_confmat': ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    }
    return metrics

# --- Training and Validation Function ---
def train_validate_fold(fold_num, train_loader, val_loader, model, criterion, optimizer, device, num_classes, idx_to_crop) -> Dict[str, float]:
    """Trains and validates one fold, returning final validation metrics."""
    print(f"\n----- Fold {fold_num + 1}/{NUM_FOLDS} -----")
    train_metrics = initialize_metrics(num_classes, device, prefix='train')
    val_metrics = initialize_metrics(num_classes, device, prefix='val')
    best_val_acc = 0.0
    fold_results = {}

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for metric in train_metrics.values(): metric.reset()

        for i, batch in enumerate(train_loader):
            # ... (data loading, forward, backward, optimizer step) ...
            pixels = batch['pixels'].to(device)
            positions = batch['positions'].to(device)
            labels = batch['label'].to(device)
            if pixels.shape[0] == 0: continue

            optimizer.zero_grad()
            outputs = model(pixels, positions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            # Update metrics - including confmat for internal state
            for metric in train_metrics.values():
                 metric.update(preds, labels)

            if (i + 1) % 100 == 0:
                print(f'  Epoch [{epoch+1}/{NUM_EPOCHS}], Fold [{fold_num+1}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss_train = running_loss / len(train_loader)
        print(f'  Epoch [{epoch+1}/{NUM_EPOCHS}] Training -> Loss: {epoch_loss_train:.4f}', end='')
        # --- Compute and log training metrics (SKIP CONFMAT PRINTING) ---
        for name, metric in train_metrics.items():
            # <<< START CHANGE >>>
            # Only compute and print scalar metrics here
            if not isinstance(metric, ConfusionMatrix):
                try:
                    value = metric.compute()
                    print(f', {name}: {value.item():.4f}', end='')
                    metric.reset() # Reset scalar metrics after computing for epoch
                except Exception as e:
                     print(f" Error computing train metric {name}: {e}", end='')
            # <<< END CHANGE >>>
        print() # Newline

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        for metric in val_metrics.values(): metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                # ... (data loading) ...
                pixels = batch['pixels'].to(device)
                positions = batch['positions'].to(device)
                labels = batch['label'].to(device)
                if pixels.shape[0] == 0: continue

                outputs = model(pixels, positions)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                # Update metrics - including confmat
                for metric in val_metrics.values():
                    metric.update(preds, labels)

        epoch_loss_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f'  Epoch [{epoch+1}/{NUM_EPOCHS}] Validation -> Loss: {epoch_loss_val:.4f}', end='')
        computed_metrics = {}
        # --- Compute and log validation metrics (Handle ConfMat separately) ---
        for name, metric in val_metrics.items():
             # <<< START CHANGE >>>
            try:
                value = metric.compute()
                if isinstance(metric, ConfusionMatrix):
                    # Store tensor but don't print it here
                    computed_metrics[name] = value
                else:
                    # Compute, print, and store scalar value
                    scalar_value = value.item()
                    print(f', {name}: {scalar_value:.4f}', end='')
                    computed_metrics[name] = scalar_value
                    # Reset scalar metrics after computing for epoch
                    metric.reset()
            except Exception as e:
                print(f" Error computing val metric {name}: {e}", end='')
             # <<< END CHANGE >>>
        print() # Newline

        # --- Best model tracking (remains the same) ---
        current_val_acc = computed_metrics.get('val_acc', 0.0)
        if current_val_acc > best_val_acc:
             best_val_acc = current_val_acc
             fold_results = computed_metrics.copy() # Store copy of best metrics
             print(f"    New best validation accuracy: {best_val_acc:.4f}")

    # --- End of fold processing (remains the same) ---
    print(f"----- Fold {fold_num + 1} Finished. Best Val Acc: {best_val_acc:.4f} -----")
    # Retrieve the confusion matrix computed over the full validation set of the BEST epoch
    conf_mat_tensor = fold_results.get('val_confmat', None)
    if conf_mat_tensor is not None:
         # Check if tensor is already on CPU, if not move it
         if conf_mat_tensor.device != torch.device('cpu'):
              conf_mat_tensor = conf_mat_tensor.cpu()
         display_confusion_matrix(conf_mat_tensor.numpy(), list(idx_to_crop.values()), fold_num + 1)
         # Don't need to reset here as metric object lifespan ends with fold

    fold_results.pop('val_confmat', None)
    return fold_results


# --- Plotting Function ---
def display_confusion_matrix(conf_matrix, class_names, fold_num):
    """Displays the confusion matrix using seaborn."""
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10))
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="viridis")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix - Fold {fold_num}')
        plt.tight_layout()
        plt.show()
    except ValueError: # Handle case with NaN values
        print(f"Error generating heatmap for fold {fold_num}, possibly due to NaN values.")
        print(df_cm)

# --- Main Execution ---
if __name__ == "__main__":
    set_random_seeds(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED}")
    print(f"Using device: {DEVICE}")

    # 1. Preprocess dataset
    print("\n--- 1. Preprocessing Dataset ---")
    (
        kept_parcel_ids,
        filtered_labels,
        crop_to_idx,
        idx_to_crop,
        num_classes,
        dates_doy,
    ) = preprocess_dataset()

    # 2. Create K-fold data loaders
    print("\n--- 2. Creating Data Loaders ---")
    fold_loaders = create_k_fold_loaders(
        parcel_ids=kept_parcel_ids,
        labels_dict=filtered_labels,
        crop_to_idx=crop_to_idx,
        data_path=DATA_PATH,
        dates_doy=dates_doy,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,
        num_pixels_sample=NUM_PIXELS_SAMPLE,
        num_workers=NUM_WORKERS,
        random_seed=RANDOM_SEED,
        max_pixel_value=PIXEL_NORMALIZATION_MAX,
    )

    # 3. Initialize Model (Outside K-Fold loop if testing architecture first)
    print("\n--- 3. Initializing Model Architecture ---")
    # Instantiate once to check parameters, will re-initialize per fold for training
    model_check = TimeSeriesTransformer(num_classes=num_classes).to(DEVICE)
    param_count = sum(p.numel() for p in model_check.parameters() if p.requires_grad)
    print(f"Model Parameter Count: {param_count:,}")
    del model_check # Remove check model

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # --- 4. Run K-Fold Training & Validation ---
    print("\n--- 4. Starting K-Fold Training & Validation ---")
    all_fold_results = [] # Store metrics dict from each fold

    if fold_loaders:
        for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
            # Re-initialize model and optimizer for each fold
            print("\nInitializing new model and optimizer for Fold", fold_idx + 1)
            model = TimeSeriesTransformer(num_classes=num_classes).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Train and validate this fold
            fold_metrics = train_validate_fold(
                fold_num=fold_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE,
                num_classes=num_classes,
                idx_to_crop=idx_to_crop
            )
            all_fold_results.append(fold_metrics)

    else:
        print("Cannot start training, no data loaders created.")

    # --- 5. Aggregate and Interpret Results ---
    print("\n--- 5. Aggregated K-Fold Results ---")
    if all_fold_results:
        # Convert list of dicts to DataFrame for easy aggregation
        results_df = pd.DataFrame(all_fold_results)
        print("\nValidation Metrics per Fold:")
        print(results_df.to_string()) # Display all fold results

        # Calculate Mean and Std Deviation across folds
        mean_metrics = results_df.mean()
        std_metrics = results_df.std()

        print("\nAverage Validation Metrics across Folds (+/- Std Dev):")
        for metric_name in mean_metrics.index:
            print(f"  {metric_name}: {mean_metrics[metric_name]:.4f} +/- {std_metrics[metric_name]:.4f}")

        print("\n--- Interpretation ---")
        print("Interpretation based on the results:")
        print("- Examine the average accuracy, weighted F1-score, and micro F1-score.")
        print("- Weighted metrics account for class imbalance, Micro metrics aggregate contributions of all classes.")
        print("- Compare Precision and Recall: High Precision means few false positives, High Recall means few false negatives.")
        print("- Analyze the confusion matrices displayed for each fold:")
        print("  - Diagonal elements show correctly classified instances per class.")
        print("  - Off-diagonal elements show misclassifications (e.g., cell [i, j] is class i predicted as class j).")
        print("  - Identify which classes are frequently confused with each other.")
        print("- Consider the standard deviation across folds: High std indicates instability or sensitivity to data splits.")
        # Add more specific interpretation based on the actual numbers you get.
        # E.g., "The model struggles particularly with class X, often confusing it with class Y, as seen..."
        # E.g., "The high weighted F1 score suggests good performance despite class imbalance."

    else:
        print("No fold results to aggregate or interpret.")

    print("\n--- Script Finished ---")