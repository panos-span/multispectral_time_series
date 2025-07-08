# Multispectral Satellite Time Series Classification

## Project Overview
This project implements a deep learning pipeline for crop type classification using multispectral satellite time series data from Sentinel-2. The goal is to classify agricultural parcels into different crop categories using temporal sequences of satellite imagery.

## Dataset
- **Source**: Timematch dataset (Denmark region, tile 32VNH, year 2017)
- **Data Type**: Multispectral satellite time series (10 spectral bands)
- **Temporal Resolution**: 52 acquisition dates per year
- **Spatial Structure**: Variable number of pixels per agricultural parcel
- **Classes**: 15 crop categories (filtered to classes with ≥200 examples)

## Architecture
- **Pixel Set Encoder (PSE)**: Processes spatial/spectral information per timestep
  - MLP1: Individual pixel processing
  - Pooling: Mean/std aggregation across pixels
  - MLP2: Feature refinement
- **Transformer Encoder**: Temporal sequence modeling
  - Sinusoidal positional encoding based on day-of-year
  - Multi-head attention mechanism
  - Classification token for sequence representation
- **MLP Classifier**: Final crop type prediction

## Key Features
- **Advanced Normalization**: Multiple strategies (z-score, min-max, robust)
- **Random Pixel Sampling**: 32 pixels per parcel during training
- **K-Fold Cross-Validation**: 5-fold evaluation for robust performance assessment
- **Fast Loading**: Preprocessed data caching for efficient iterations

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-score (micro and weighted averages)
- Confusion matrices for detailed class-wise analysis
- Cross-validation statistics with mean ± standard deviation
