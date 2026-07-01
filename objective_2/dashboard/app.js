// Globals
const API_BASE = 'http://localhost:8000/api';
let dates = [];
let currentDateIdx = 0;
let mapDataCache = {};
let chartInstance = null;

// UI Elements
const slider = document.getElementById('date-slider');
const dateDisplay = document.getElementById('current-date-display');
const loadingInd = document.getElementById('loading-indicator');
const noDataBanner = document.getElementById('no-data-banner');
const tooltip = document.getElementById('tooltip');
const regionSelect = document.getElementById('region-select');

// Layer Toggles
const toggles = {
    hcho: document.getElementById('layer-hcho'),
    fires: document.getElementById('layer-fires'),
    hotspots: document.getElementById('layer-hotspots'),
    disagree: document.getElementById('layer-disagree'),
    dbscan: document.getElementById('layer-dbscan'),
    uncertainty: document.getElementById('layer-uncertainty')
};

// Deck.gl Setup
const deckgl = new deck.DeckGL({
    container: 'map-container',
    initialViewState: {
        longitude: 78.0,
        latitude: 22.0,
        zoom: 4.5,
        pitch: 0,
        bearing: 0
    },
    controller: true,
    getTooltip: ({ object }) => {
        if (!object) return null;
        let html = `<b>HCHO:</b> ${object.hcho === -999 ? 'No Data' : object.hcho.toExponential(3)}<br/>`;
        html += `<b>Fires:</b> ${object.fires}<br/>`;
        if (toggles.hotspots.checked) html += `<b>Consensus Hotspot:</b> ${object.consensus === 1 ? 'Yes' : 'No'}<br/>`;
        if (toggles.disagree.checked) html += `<b>Disagreement (A vs C):</b> ${object.disagreement === 1 ? 'Yes' : 'No'}<br/>`;
        if (toggles.dbscan.checked) html += `<b>Cluster ID:</b> ${object.cluster !== -1 ? object.cluster : 'None'}<br/>`;
        if (object.uncertainty === 1) html += `<b>Uncertainty:</b> High (Cloud Fill)`;
        return { html: html };
    }
});

// Region Zoom logic
const regions = {
    'india': { longitude: 78.0, latitude: 22.0, zoom: 4.5 },
    'punjab': { longitude: 75.5, latitude: 30.5, zoom: 7 },
    'up': { longitude: 78.5, latitude: 28.0, zoom: 6.5 },
    'central': { longitude: 80.0, latitude: 20.0, zoom: 6 }
};

regionSelect.addEventListener('change', (e) => {
    const r = regions[e.target.value];
    deckgl.setProps({
        initialViewState: {
            ...r,
            transitionDuration: 1000,
            transitionInterpolator: new deck.FlyToInterpolator()
        }
    });
});

// Color scales
function getHCHOColor(val) {
    if (val === -999) return [0, 0, 0, 0];
    const normalized = Math.min(Math.max((val - 0.0001) / 0.0003, 0), 1);
    // Dark blue to bright red
    return [Math.floor(255 * normalized), Math.floor(50 * (1 - normalized)), Math.floor(255 * (1 - normalized)), 180];
}

const clusterColors = [
    [255, 0, 0], [0, 255, 0], [0, 100, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]
];

// Update Layers
function updateLayers(data) {
    if (!data) {
        deckgl.setProps({ layers: [] });
        return;
    }

    const layers = [];

    // Basemap using Carto Voyager (Soft Light) via TileLayer
    layers.push(new deck.TileLayer({
        data: 'https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
        minZoom: 0,
        maxZoom: 19,
        renderSubLayers: props => {
            const { boundingBox } = props.tile;
            return new deck.BitmapLayer(props, {
                data: null,
                image: props.data,
                bounds: [boundingBox[0][0], boundingBox[0][1], boundingBox[1][0], boundingBox[1][1]]
            });
        }
    }));

    // Reconstruct polygon array from parallel arrays
    const gridData = [];
    for (let i = 0; i < data.lats.length; i++) {
        const lat = data.lats[i];
        const lon = data.lons[i];
        gridData.push({
            polygon: [
                [lon - 0.125, lat - 0.125],
                [lon + 0.125, lat - 0.125],
                [lon + 0.125, lat + 0.125],
                [lon - 0.125, lat + 0.125]
            ],
            hcho: data.hcho[i],
            fires: data.fires[i],
            consensus: data.consensus[i],
            cluster: data.cluster[i],
            disagreement: data.disagreement[i],
            uncertainty: data.uncertainty[i]
        });
    }

    // Grid Layer (HCHO, Clusters, Disagreement, Uncertainty)
    layers.push(new deck.SolidPolygonLayer({
        id: 'hcho-grid',
        data: gridData,
        getPolygon: d => d.polygon,
        getFillColor: d => {
            if (toggles.disagree.checked && d.disagreement) return [255, 100, 0, 200];
            if (toggles.dbscan.checked && d.cluster !== -1) return clusterColors[d.cluster % clusterColors.length].concat([200]);
            if (toggles.hotspots.checked && d.consensus) return [255, 0, 0, 150];

            // Apply uncertainty dimming
            if (toggles.uncertainty.checked && d.uncertainty === 1) {
                return [50, 50, 50, 100]; // Dark clouds
            }

            if (toggles.hcho && toggles.hcho.checked) {
                return getHCHOColor(d.hcho);
            }
            return [0, 0, 0, 0];
        },
        pickable: true,
        updateTriggers: {
            getFillColor: [
                toggles.hcho ? toggles.hcho.checked : true,
                toggles.disagree.checked,
                toggles.dbscan.checked,
                toggles.hotspots.checked,
                toggles.uncertainty.checked
            ]
        }
    }));

    // Fire Layer
    if (toggles.fires.checked) {
        layers.push(new deck.ScatterplotLayer({
            id: 'fire-dots',
            data: gridData.filter(d => d.fires > 0),
            getPosition: d => [(d.polygon[0][0] + d.polygon[2][0]) / 2, (d.polygon[0][1] + d.polygon[2][1]) / 2],
            getFillColor: [255, 150, 0, 255],
            getRadius: d => Math.log(d.fires + 1) * 3000,
            pickable: false,
            stroked: true,
            getLineColor: [255, 255, 255, 200],
            lineWidthMinPixels: 1
        }));
    }

    // Actual Geometric India Boundary (Matches Matplotlib)
    layers.push(new deck.GeoJsonLayer({
        id: 'india-boundaries',
        data: 'india.json',
        stroked: true,
        filled: false,
        lineWidthMinPixels: 1.5,
        getLineColor: [0, 0, 0, 220], // Sharp black border
        pickable: false
    }));

    deckgl.setProps({ layers });
}

// Fetch Map Data
let fetchAbortController = null;

async function fetchMapData(date, prefetch = false) {
    if (mapDataCache[date]) {
        if (!prefetch) {
            if (mapDataCache[date] === 'no_data') {
                noDataBanner.classList.remove('hidden');
                updateLayers(null);
            } else {
                noDataBanner.classList.add('hidden');
                updateLayers(mapDataCache[date]);
            }
            loadingInd.classList.add('hidden');
        }
        return;
    }

    if (!prefetch) loadingInd.classList.remove('hidden');

    try {
        const response = await fetch(`${API_BASE}/map?date=${date}`);
        const result = await response.json();

        if (result.status === 'no_data') {
            mapDataCache[date] = 'no_data';
            if (!prefetch) {
                noDataBanner.classList.remove('hidden');
                updateLayers(null);
            }
        } else {
            mapDataCache[date] = result.data;
            if (!prefetch) {
                noDataBanner.classList.add('hidden');
                updateLayers(result.data);
            }
        }
    } catch (e) {
        console.error("Failed to load map data", e);
    } finally {
        if (!prefetch) loadingInd.classList.add('hidden');
    }
}

// Slider Debounce & Prefetch Logic
let sliderTimeout;
slider.addEventListener('input', (e) => {
    currentDateIdx = parseInt(e.target.value);
    const date = dates[currentDateIdx];
    dateDisplay.innerText = date;

    // Highlight chart
    if (chartInstance) {
        chartInstance.setActiveElements([{ datasetIndex: 0, index: currentDateIdx }]);
        chartInstance.update();
    }

    clearTimeout(sliderTimeout);
    sliderTimeout = setTimeout(() => {
        fetchMapData(date);

        // Prefetch adjacent days
        if (currentDateIdx < dates.length - 1) fetchMapData(dates[currentDateIdx + 1], true);
        if (currentDateIdx > 0) fetchMapData(dates[currentDateIdx - 1], true);

    }, 150); // 150ms debounce
});

// Wire up checkboxes to force re-render
Object.values(toggles).forEach(toggle => {
    toggle.addEventListener('change', () => {
        const date = dates[currentDateIdx];
        if (mapDataCache[date] && mapDataCache[date] !== 'no_data') {
            updateLayers(mapDataCache[date]);
        }
    });
});

// Load Initial Data
async function init() {
    // 1. Fetch dates
    const res = await fetch(`${API_BASE}/dates`);
    const d = await res.json();
    dates = d.dates;

    slider.max = dates.length - 1;

    // Default to Nov 5 (peak burning)
    currentDateIdx = dates.indexOf('2023-11-05');
    if (currentDateIdx === -1) currentDateIdx = 0;

    slider.value = currentDateIdx;
    dateDisplay.innerText = dates[currentDateIdx];

    // 2. Fetch Chart Data
    const cRes = await fetch(`${API_BASE}/timeseries`);
    const cData = await cRes.json();

    const ctx = document.getElementById('igp-chart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: cData.dates,
            datasets: [
                {
                    label: 'HCHO (IGP Mean)',
                    data: cData.hcho,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    yAxisID: 'y',
                    tension: 0.1
                },
                {
                    label: 'Fire Count (IGP)',
                    data: cData.fires,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    yAxisID: 'y1',
                    type: 'bar',
                    backgroundColor: 'rgba(255, 99, 132, 0.5)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { type: 'linear', display: true, position: 'left' },
                y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } }
            },
            interaction: { mode: 'index', intersect: false },
            onHover: (e, activeEls) => {
                if (activeEls.length > 0) {
                    const idx = activeEls[0].index;
                    if (idx !== currentDateIdx) {
                        slider.value = idx;
                        slider.dispatchEvent(new Event('input'));
                    }
                }
            }
        }
    });

    // 3. Load initial map
    fetchMapData(dates[currentDateIdx]);
}

init();
