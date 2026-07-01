# Model B: Physics-Informed ConvLSTM

**Status:** Architecture Completed. Awaiting 4-Year Data Stack.

## Overview

Model B represents our team's full scientific vision for spatio-temporal Air Quality Index (AQI) prediction. While **Model A (CNN+XGBoost)** provided highly accurate results over a limited 4-month window, it functioned purely as a data-driven model without intrinsic temporal memory or physical constraints.

To address these limitations, we developed **Model B**, a sophisticated **Physics-Informed Convolutional LSTM (ConvLSTM)** designed to operate on a massive 4-year dataset (2020–2023).

## Key Architectural Innovations

### 1. Spatio-Temporal Memory (14-Day Lookback)
Unlike static CNNs, atmospheric pollution is highly dependent on transport over time. This architecture utilizes a 14-day temporal sequence window, allowing the model to track the movement, diffusion, and accumulation of pollution plumes (like crop-fire smoke or industrial emissions) across the Indian subcontinent over two weeks.

### 2. Physics-Informed Loss Function
The core innovation of this model is the integration of an **Advection-Diffusion PDE (Partial Differential Equation)** directly into the training objective. 
Instead of relying solely on Mean Squared Error (MSE) against ground-truth stations, the loss function actively penalizes the network if its spatial predictions violate the laws of atmospheric transport, guided by ERA5 wind vectors ($U, V$).

### 3. Adaptive BMA (Bayesian Model Averaging)
Because emissions-driven pollutants (like NO₂ and SO₂) behave differently from transport-driven pollutants (like PM2.5 and CO), the pipeline includes an adaptive Bayesian Model Averaging hook to intelligently weight the ConvLSTM's output against localized XGBoost predictions.

## Limitations & Future Work
As documented in our technical report, the architecture is fully built and structurally tested (see the Jupyter Notebook in this directory). However, training is pending the final assembly of the 4-year, 18-channel `.npy` data stack. 

Immediate next steps include:
- Completing the extraction and regridding of 2020-2022 Sentinel-5P data.
- Integrating a dynamic emission source term ($S$) into the PDE loss for improved localized accuracy.
- Training the model end-to-end on high-performance GPU clusters.
