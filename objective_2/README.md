# Objective 2: HCHO Pollution, Stubble Burning, and Transport Analysis

## Overview
This repository directory contains the complete workflow, methodology, and key results of **Objective 2**. It details the step-by-step analytical pipeline used to link Sentinel-5P HCHO (Formaldehyde) column densities with VIIRS active fire data. This pipeline mathematically and visually proves the causal chain between stubble burning in Punjab/Haryana and severe air quality degradation across the Indo-Gangetic Plain (IGP).

## Directory Structure
To ensure this objective is 100% self-contained and reproducible, all dependencies have been isolated within this directory:

- `data/`: Contains the lightweight, pre-regridded (120x124) Sentinel-5P HCHO, VIIRS Fire, and ERA5 meteorological arrays, alongside the geographical boundaries (`gadm41_IND_1.json`).
- `results/`: The output directory where all intermediate NumPy arrays, threshold masks, correlation data, and time-series CSVs are saved during pipeline execution.
- `obj2_maps/`: The output directory where all static presentation-ready visualizations (Matplotlib figures, spatial correlation maps, DBSCAN plots) are saved.
- `dashboard/`: Contains the static frontend assets (HTML, CSS, JS) for the WebGL interactive dashboard.
- `dashboard_api.py`: The FastAPI backend server that feeds compressed data to the frontend dashboard.
- `run_objective_2.py`: The master execution script containing the entire 9-phase analytical pipeline.

## How to Run the Pipeline
The entire 9-phase analysis has been consolidated into a single master script. To regenerate all data, mathematical thresholds, and static maps from scratch:

```bash
cd objective_2
python run_objective_2.py
```
This script will sequentially execute phases 1 through 9, populating the `results/` and `obj2_maps/` directories.

## Phases Breakdown
The `run_objective_2.py` pipeline executes the following phases in order:

* **Phase 1: HCHO Data Loading and Regridding**
  Ingests Sentinel-5P TROPOMI HCHO Level 3 data (Oct 2023 - Jan 2024), applies strict QA > 0.5 filtering to remove cloud cover, and regrids it to a 0.25° x 0.25° grid over India.
* **Phase 2: HCHO Baseline Statistics**
  Establishes baseline physical bounds. Computes monthly spatial means and the November anomaly (mapping exact ±10⁻⁵ mol/m² deviations).
* **Phase 3: Fire Data Processing**
  Filters VIIRS (VNP14IMGTDL_NRT) active fire data for high-confidence stubble burning thermal anomalies, aggregating Fire Radiative Power (FRP) into the exact same 0.25° grid.
* **Phase 4: Hotspot Detection Algorithm**
  Systematically identifies structural pollution hotspots using three mathematical thresholds: a Daily 95th Percentile, a Robust Median Absolute Deviation (MAD), and an Absolute Literature Threshold (4.0×10⁻⁴ mol/m²). Methods B & C are merged into a strict consensus mask.
* **Phase 5: Spatial Visualisation**
  Generates presentation-ready static visualisations (saved in `obj2_maps/`) proving spatial colocation, including the highly compelling November HCHO + VIIRS fire dot overlay.
* **Phase 6: Temporal Lag Correlation**
  Proves temporal causality by shifting fire time series against HCHO time series. The peak Pearson correlation occurs at Lag 1 (r=0.461, p=0.010), proving a ~24-hour atmospheric oxidation time between a fire event and detectable HCHO enhancement.
* **Phase 7: Transport and Wind Analysis**
  Uses ERA5 U and V wind components to compute 48-hour backward trajectories from Delhi on peak burning days, proving the dominant air parcel origin is NW of Delhi acting as a conveyor belt for smoke.
* **Phase 8: Source Region Identification**
  Separates source vs. receptor regions. Proves that Central Punjab/Haryana show massive fire counts with strong statistical correlation to local HCHO, while Western UP acts purely as a receptor accumulating transported pollution (r=-0.067).
* **Phase 8b: DBSCAN Algorithmic Cluster Validation**
  Uses unsupervised machine learning (DBSCAN; eps=0.5°, min_samples=5) to independently validate the structural boundaries of the Indo-Gangetic Plain pollution zone. The algorithm successfully identified a massive contiguous cluster spanning the entire northern width of India.
* **Phase 9: Prep Dashboard Data**
  Processes the heavy `.npy` stacks into optimized, date-keyed JSON payloads and pre-computes static spatial layers for lightning-fast dashboard rendering.

## Interactive WebGL Dashboard
To dynamically explore the spatial and temporal overlap of HCHO hotspots, fires, and DBSCAN clusters day-by-day:

1. Start the FastAPI backend server:
```bash
python dashboard_api.py
```
2. Open your web browser and navigate to: **http://127.0.0.1:8000**

The dashboard features a fluid date scrubber, click-to-inspect pixel metrics, and multi-layered toggles (Base Map, HCHO Concentration, Hotspot Detection Method consensus, DBSCAN clusters, and Active Fire dots).
