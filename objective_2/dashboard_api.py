import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add GZIP compression (min size 1KB) to compress parallel arrays
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = 'results'
GRID_SPEC = 'data/regridded/grid_spec.json'

print("Loading arrays into memory...")
HCHO_stack = np.load(os.path.join(RESULTS_DIR, 'HCHO_daily_stack.npy'))
fire_stack = np.load(os.path.join(RESULTS_DIR, 'fire_counts_daily.npy'))
consensus  = np.load(os.path.join(RESULTS_DIR, 'hotspot_consensus.npy'))
dbscan     = np.load(os.path.join(RESULTS_DIR, 'dbscan_grid.npy'))
disagree   = np.load(os.path.join(RESULTS_DIR, 'hotspot_disagreement.npy'))
uncertainty= np.load(os.path.join(RESULTS_DIR, 'uncertainty_mask.npy'))

with open(os.path.join(RESULTS_DIR, 'HCHO_dates.json')) as f:
    dates = json.load(f)

# Load Lat/Lon grids
with open(GRID_SPEC) as f:
    gs = json.load(f)
lats_1d = np.linspace(gs['lat_max'], gs['lat_min'], gs['nlat'])
lons_1d = np.linspace(gs['lon_min'], gs['lon_max'], gs['nlon'])
lons_grid, lats_grid = np.meshgrid(lons_1d, lats_1d)

print("Backend API ready.")

@app.get("/api/map")
def get_map_data(date: str = Query(..., description="Date in YYYY-MM-DD")):
    if date not in dates:
        return JSONResponse(content={"status": "no_data", "message": "Date out of range"})
    
    idx = dates.index(date)
    hcho_day = HCHO_stack[idx]
    fire_day = fire_stack[idx]
    unc_day  = uncertainty[idx]
    
    # Valid pixels are those where we have ANY data (either HCHO or Fires)
    # Even if HCHO is NaN (cloudy), if there's a fire we might want it?
    # Usually we just base it on HCHO valid mask, but let's include all 
    # to be safe, except where BOTH are nan/0. Actually, the grid is dense enough
    # that we can just return all 14,880 pixels and let frontend handle it, 
    # but filtering cuts size. Let's filter out NaN HCHO pixels that have 0 fires.
    
    # Wait, the user asked for a "No Data - Cloud cover" banner if the whole day is missing.
    valid_mask = ~np.isnan(hcho_day) | (fire_day > 0)
    
    if not np.any(valid_mask):
        return JSONResponse(content={"status": "no_data"})
    
    # Replace NaNs with -999 for JSON serialization
    hcho_safe = np.where(np.isnan(hcho_day), -999, hcho_day)
    
    # Extract data for valid pixels
    lats_out = lats_grid[valid_mask].round(3).tolist()
    lons_out = lons_grid[valid_mask].round(3).tolist()
    hcho_out = hcho_safe[valid_mask].tolist()
    fire_out = fire_day[valid_mask].tolist()
    cons_out = consensus[valid_mask].astype(int).tolist()
    clus_out = dbscan[valid_mask].tolist()
    disg_out = disagree[valid_mask].astype(int).tolist()
    unc_out  = unc_day[valid_mask].astype(int).tolist()
    
    return JSONResponse(content={
        "status": "success",
        "data": {
            "lats": lats_out,
            "lons": lons_out,
            "hcho": hcho_out,
            "fires": fire_out,
            "consensus": cons_out,
            "cluster": clus_out,
            "disagreement": disg_out,
            "uncertainty": unc_out
        }
    })

@app.get("/api/timeseries")
def get_timeseries():
    ts_file = os.path.join(RESULTS_DIR, 'HCHO_igp_timeseries.csv')
    df = pd.read_csv(ts_file)
    
    fire_file = os.path.join(RESULTS_DIR, 'fire_igp_timeseries.csv')
    df_fire = pd.read_csv(fire_file)
    
    merged = pd.merge(df, df_fire, on='date', how='outer')
    merged = merged.fillna(0)
    
    return JSONResponse(content={
        "dates": merged['date'].tolist(),
        "hcho": merged['hcho_igp_mean'].tolist(),
        "fires": merged['fire_count_total'].tolist()
    })

@app.get("/api/dates")
def get_dates():
    return JSONResponse(content={"dates": dates})

os.makedirs("dashboard", exist_ok=True)
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    print("Starting dashboard server...")
    uvicorn.run("dashboard_api:app", host="127.0.0.1", port=8000, reload=True)
