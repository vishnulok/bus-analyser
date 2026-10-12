# APSRTC AI Demand Forecasting – Complete Feature & Function Reference

> **Project:** APSRTC AI Demand Forecasting & Dynamic Bus Allocation  
> **Stack:** Python · Flask · scikit-learn · Jinja2 · Plotly · Vanilla JS  
> **Entry point:** pp.py  
> **Model file:** psrtc_demand_model.pkl (Extra Trees Regressor pipeline)

---

## Table of Contents

1. [Pages & Routes](#pages--routes)
2. [Backend Functions (app.py)](#backend-functions-apppy)
3. [Machine Learning Model](#machine-learning-model)
4. [REST API Endpoints](#rest-api-endpoints)
5. [Static Reference Data](#static-reference-data)
6. [Frontend Pages & Features](#frontend-pages--features)
7. [Retrain Pipeline (retrain_model.py)](#retrain-pipeline-retrain_modelpy)

---

## Pages & Routes

| URL | View Function | Template | Description |
|-----|--------------|----------|-------------|
| / | index() | index.html | Landing page with hero, model metrics, feature overview |
| /prediction | prediction_page() | prediction.html | Single-trip demand prediction form |
| /optimization | optimization_page() | optimization.html | Fleet-wide dynamic bus allocation |
| /dashboard | dashboard_page() | dashboard.html | Live urban mobility dashboard |
| /explore | explore_page() | exploration.html | Exploratory data analysis & route statistics |

---

## Backend Functions (app.py)

### _season_of_month(month: int) -> str
Maps a calendar month number to one of four season labels used during model training.

| Month Range | Season Label |
|------------|-------------|
| Dec, Jan, Feb | "winter" |
| Mar, Apr, May | "summer" |
| Jun, Jul, Aug, Sep | "monsoon" |
| Oct, Nov | "post_monsoon" |

---

### uild_features(...) -> pd.DataFrame
Constructs the 20-feature DataFrame required by the ML pipeline.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| oute | str | Route name (e.g., "Hyderabad-Vijayawada") |
| us_type | str | Bus type (e.g., "Express", "Volvo AC") |
| depot | str | Depot name (e.g., "Hyderabad") |
| capacity | int | Seating capacity of the bus |
| distance_km | float | Route distance in km |
| are_per_passenger | float | Ticket price in INR |
| date_obj | datetime | Travel date |
| previous_passengers | float|None | Previous trip passenger count |
| previous_3_avg | float|None | 3-day rolling average passengers |
| previous_5_avg | float|None | 5-day rolling average passengers |
| previous_7_avg | float|None | 7-day rolling average passengers |
| oute_avg_demand | float|None | Historical route-level average |

**Engineered Features Generated:**

| Feature | Formula |
|---------|---------|
| is_weekend | 1 if weekday >= 5 (Sat/Sun) |
| is_holiday | 1 if (month, day) in AP holiday calendar |
| quarter | (month - 1) // 3 + 1 |
| season | From _season_of_month() |
| evenue_per_km | are_per_passenger / distance_km |
| demand_score | ase_demand × frequency_multiplier / 100 |
| day_of_year | 1–366 |
| week_of_year | 0–53 |

---

### demand_level(occupancy_pct: float) -> str
Classifies predicted occupancy into a demand level label.

| Occupancy % | Label |
|------------|-------|
| ≥ 80% | "HIGH" |
| 50–79% | "MEDIUM" |
| < 50% | "LOW" |

---

### ecommended_buses(predicted_passengers: int, capacity: int) -> int
Calculates minimum buses needed to cover demand.

`python
return max(1, math.ceil(predicted_passengers / capacity))
`

---

### explore_page() -> Response
Computes live summary statistics from ROUTES_CONFIG and passes them to exploration.html.

**Variables passed to template:**

| Variable | Type | Description |
|----------|------|-------------|
| summary.routes | int | Total number of active routes |
| summary.avg_demand | float | Average base demand across all routes |
| summary.avg_distance | float | Average route distance in km |
| summary.max_demand | int | Highest base demand value |
| summary.max_route | str | Route with highest base demand |
| summary.min_demand | int | Lowest base demand value |
| summary.min_route | str | Route with lowest base demand |
| summary.bus_types | int | Count of distinct bus types |
| summary.depots | int | Count of operating depots |
| outes | list[dict] | Full ROUTES_CONFIG list |

---

## Machine Learning Model

### Model File: psrtc_demand_model.pkl

Serialized scikit-learn pipeline (joblib). Loaded once at startup.

### Training Script: etrain_model.py

Four benchmark models are trained:

| Model | Algorithm | R² | MAE | RMSE |
|-------|-----------|-----|-----|------|
| ExtraTrees | ExtraTreesRegressor (300 trees, max_depth=18) | ~0.90 | ~1.31 | ~1.66 |
| HistGradientBoosting | HistGradientBoostingRegressor | ~0.87 | ~1.55 | ~1.89 |
| RandomForest | RandomForestRegressor (200 trees) | ~0.86 | ~1.63 | ~1.97 |
| GradientBoosting | GradientBoostingRegressor | ~0.84 | ~1.82 | ~2.20 |

The **ExtraTrees** model is used in production (highest R²).

### Feature Input Order (20 features)
`
route, bus_type, depot, capacity, distance_km, fare_per_passenger,
month, day_of_week, day_of_year, week_of_year,
previous_passengers, previous_3_avg, previous_5_avg, previous_7_avg,
route_avg_demand, is_weekend, is_holiday, quarter, season,
revenue_per_km, demand_score
`

### AP Holiday Calendar (_AP_HOLIDAYS)
Covers all major Andhra Pradesh public holidays:
Jan 1, Jan 14 (Pongal), Jan 26 (Republic Day), Mar 17-18 (Holi),
Apr 1, Apr 14 (Ambedkar Jayanti), Apr 21, May 1 (Labour Day),
Jun 17, Aug 15 (Independence Day), Sep 5-7 (Ganesh Chaturthi),
Oct 2 (Gandhi Jayanti), Oct 12-14 (Dasara), Oct 24, Nov 5, Dec 25.

---

## REST API Endpoints

### POST /api/predict
**Purpose:** Predict passenger demand for a single trip.

**Request Body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| oute | string | "Hyderabad-Vijayawada" | Route name |
| us_type | string | "Express" | Bus type |
| depot | string | "Hyderabad" | Depot |
| capacity | int | 52 | Seating capacity |
| distance_km | float | 275 | Route distance |
| are_per_passenger | float | 300 | Fare (₹) |
| date | string | today | Date in YYYY-MM-DD |
| previous_passengers | float | null | Prior trip count |
| previous_3_avg | float | null | 3-day avg |
| previous_5_avg | float | null | 5-day avg |
| previous_7_avg | float | null | 7-day avg |
| oute_avg_demand | float | null | Route historical avg |

**Response (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| success | bool | Request status |
| predicted_passengers | int | ML-predicted passenger count |
| occupancy_pct | float | % of seats filled |
| demand_level | string | HIGH / MEDIUM / LOW |
| ecommended_buses | int | Buses needed to cover demand |
| daily_frequency | string | Human-readable schedule (e.g., "Every 30 mins (16 departures/day)") |
| estimated_revenue | float | predicted_passengers × fare (₹) |

---

### POST /api/optimize
**Purpose:** Allocate a fixed fleet across all 15 routes using the Greedy Demand-Pressure Algorithm.

**Request Body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| 	otal_buses | int | 30 | Total fleet size to distribute |
| date | string | today | Date in YYYY-MM-DD |
| us_type | string | "Express" | Bus type applied to all routes |

**Algorithm – Greedy Demand-Pressure:**
1. Assign **1 bus minimum** to each route (if fleet allows).
2. Compute demand pressure: 	otal_corridor_demand / (allocated_buses × capacity × 2).
3. Assign the next available bus to the route with **highest pressure**.
4. Repeat until fleet is exhausted.

**Per-route Response Fields:**

| Field | Description |
|-------|-------------|
| oute | Route name |
| single_trip_predicted | ML prediction for one trip |
| predicted_passengers | Total daily demand (single_trip × freq_multiplier) |
| daily_frequency | Human-readable departure schedule |
| corridor_type | Corridor classification (e.g., "High-Density Express Corridor") |
| llocated_buses | Buses allocated by greedy algorithm |
| llocated_capacity | uses × capacity × 2 seats/day |
| utilization | Demand as % of allocated capacity |
| demand_level | HIGH / MEDIUM / LOW |
| llocation_reason | Plain-English explanation of why N buses were allocated |
| ecommendation | Short deployment instruction |
| priority_rank | 1 = highest demand route |

---

### GET /api/dashboard
**Purpose:** Returns full optimization result for dashboard display.

**Query Parameters:** date, 	otal_buses, us_type

Internally calls pi_optimize() and returns identical JSON structure.

---

### GET /api/routes
Returns the full ROUTES_CONFIG list (all 15 routes with distance, fare, frequency, corridor type).

---

### GET /api/bus_types
Returns available bus types and their seating capacities.

`json
{
  "bus_types": ["Express","Ordinary","Semi-Sleeper","Super Luxury","Sleeper","Volvo AC"],
  "capacities": {"Ordinary":55,"Express":52,"Semi-Sleeper":48,"Super Luxury":44,"Volvo AC":42,"Sleeper":36}
}
`

---

## Static Reference Data

### ROUTES_CONFIG (15 routes)

| Route | Depot | Distance | Fare | Base Demand | Frequency |
|-------|-------|----------|------|-------------|-----------|
| Hyderabad-Vijayawada | Hyderabad | 275 km | ₹300 | 190 | Every 30 mins (16/day) |
| Hyderabad-Visakhapatnam | Hyderabad | 620 km | ₹650 | 210 | Every 45 mins (12/day) |
| Hyderabad-Tirupati | Hyderabad | 560 km | ₹600 | 175 | Every 45 mins (10/day) |
| Guntur-Hyderabad | Guntur | 285 km | ₹310 | 160 | Every 40 mins (14/day) |
| Vijayawada-Visakhapatnam | Vijayawada | 350 km | ₹380 | 180 | Every 40 mins (12/day) |
| Nellore-Chennai | Nellore | 175 km | ₹190 | 170 | Every 30 mins (18/day) |
| Rajahmundry-Hyderabad | Visakhapatnam | 430 km | ₹460 | 165 | Every 1 hr (8/day) |
| Vijayawada-Tirupati | Vijayawada | 380 km | ₹400 | 155 | Every 1 hr (8/day) |
| Ongole-Hyderabad | Guntur | 310 km | ₹330 | 150 | Every 1.5 hrs (8/day) |
| Kurnool-Hyderabad | Kurnool | 215 km | ₹240 | 145 | Every 1 hr (10/day) |
| Eluru-Hyderabad | Vijayawada | 330 km | ₹350 | 140 | Every 2 hrs (6/day) |
| Anantapur-Bangalore | Kurnool | 215 km | ₹230 | 135 | Every 1 hr (10/day) |
| Kadapa-Hyderabad | Hyderabad | 410 km | ₹450 | 130 | Every 2 hrs (6/day) |
| Kakinada-Vijayawada | Vijayawada | 210 km | ₹230 | 125 | Every 1 hr (10/day) |
| Chittoor-Bangalore | Tirupati | 180 km | ₹190 | 120 | Every 1.5 hrs (8/day) |

### BUS_CAPACITIES

| Type | Seats |
|------|-------|
| Ordinary | 55 |
| Express | 52 |
| Semi-Sleeper | 48 |
| Super Luxury | 44 |
| Volvo AC | 42 |
| Sleeper | 36 |

### _ROUTE_BASE_OCCUPANCY
Historical average occupancy per route (range: 0.52 – 0.82). Used to initialise oute_avg_demand when no prior data is provided.

---

## Frontend Pages & Features

### index.html – Landing Page
- Animated hero with gradient mesh background and CSS grid overlay
- **Improvement banner**: shows model R² upgrade (0.72 → 0.90)
- **Hero stats**: 92% Prediction Accuracy, 57% Overcrowding Reduction
- **6 insight cards**: Routes monitored, model accuracy, bus types, time periods, depots, data samples
- **3 feature cards**: AI Demand Prediction, Dynamic Fleet Allocation, Real-time Dashboard
- **Model Comparison Table** – 4 models with R², MAE, RMSE
- **Technology badges**: Python, Flask, scikit-learn, Jinja2, Plotly, JS
- Navigation to all 5 pages

### prediction.html – Demand Prediction
- Route selector → auto-fills **Route Distance** and **Seating Capacity** (read-only)
- Bus type selector → auto-fills capacity
- Date picker
- Optional: Previous passenger counts (1-day, 3-day avg, 5-day avg, 7-day avg)
- Calls POST /api/predict
- Result card: predicted passengers, occupancy %, demand level badge, recommended buses, estimated revenue, daily frequency
- 3 operational scenario presets (Peak, Holiday, Low-demand)

### optimization.html – Fleet Optimization
- Fleet size input (total buses)
- Bus type selector
- Date picker
- Calls POST /api/optimize
- Summary bar: total routes, avg demand, avg utilization, bus type
- Per-route cards showing:
  - Priority rank badge
  - Predicted passengers (per trip + daily total)
  - Bus frequency schedule
  - Allocated buses with reason
  - Utilization bar + demand level badge
  - Allocation reason (plain English)

### dashboard.html – Live Dashboard
- Auto-refreshes every 60 seconds via GET /api/dashboard
- Summary KPI row: total routes, total predicted demand, buses deployed, avg utilization
- Corridor cards with occupancy meters and frequency badges
- Demand level colour coding (green HIGH / orange MEDIUM / red LOW)

### exploration.html – Exploratory Analysis *(new)*
- **6 KPI cards**: Total Routes, Avg Demand, Avg Distance, Peak Demand, Bus Types, Min Demand
- **Demand Bar Chart** (Plotly horizontal bar, colour-coded by demand level)
- **Distance Donut Chart** (Short / Medium / Long route bands)
- **Demand vs. Distance Scatter Plot** (bubble size = fare, colour = frequency multiplier)
- **Route Summary Table**: all 15 routes with occupancy mini-bars and High/Medium/Low badges

---

## Retrain Pipeline (retrain_model.py)

| Step | Description |
|------|-------------|
| Data Generation | Synthetic dataset with seasonal/temporal/route features |
| Feature Engineering | is_weekend, is_holiday, quarter, season, evenue_per_km, demand_score |
| Column Transformer | OrdinalEncoder for categorical, StandardScaler + PowerTransformer for numeric |
| Model Training | 4 models with different configs; best model saved to psrtc_demand_model.pkl |
| Compression | joblib.dump(model, path, compress=6) |
| Benchmark Output | R², MAE, RMSE for all 4 models printed to console |
