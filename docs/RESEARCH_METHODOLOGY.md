# Research Methodology — Cylon Armada

## What the Experiments Measure and Why

---

## 1. The Core Research Question

> **Can a distributed serverless HPC framework reduce LLM inference cost and latency
> through context reuse, and does peer-to-peer communication (FMI/TCPunch) outperform
> a Redis intermediary as the mechanism for sharing that context?**

This is an empirical systems research question. The experiments **create a controlled
distributed AI workload and measure** four quantities across varying degrees of parallelism:

1. **Cache hit rate** — what fraction of LLM calls were avoided by reusing a prior response
2. **Average task latency** — time per individual inference request (ms)
3. **Wall-clock time** — total workflow completion time across all parallel workers (ms)
4. **Cost** — actual USD spent on Bedrock API calls vs. the counterfactual baseline cost

---

## 2. What Is Being Created

The framework (`cylon-armada`) is a novel contribution — not an off-the-shelf tool:

- A **serverless parallel execution engine** using AWS Step Functions Map state to dispatch
  N concurrent Lambda workers (`armada_executor`) from a single workflow invocation
- A **context-reuse router** inside each worker that embeds the task, searches prior
  responses via SIMD cosine similarity, and returns a cached answer or calls the LLM
- A **peer-to-peer context broadcast** layer using Apache Cylon FMI over TCPunch-established
  direct TCP connections between Lambda workers
- An **AstroMAE inference pipeline** (Cosmic AI) that feeds real astronomical observations
  into the LLM task generation pipeline, connecting deep learning inference to the
  context-reuse framework
- Python and Node.js runtime variants of all components for cross-language comparison

---

## 3. What the Similarity Search Compares

**Precise answer: what are we running similarity against?**

Each `armada_executor` worker holds a query embedding for its assigned task. It searches
that query against the **corpus of embeddings for all previously completed LLM responses**
stored in the shared context store (Redis or Cylon ContextTable) for the same `workflow_id`.

```
Query:   embedding(current_task_description)            ← 1024-dim float32 vector
Corpus:  { embedding(prior_task_i) : i ∈ completed }   ← all prior responses this workflow
```

**Step by step:**

1. `armada_init` embeds every task description using **Amazon Titan Embed v2**
   (1024-dimensional, L2-normalized float32 vectors). Stored in Redis as
   `embedding:{workflow_id}:{rank}`.

2. Each `armada_executor` fetches its query embedding from Redis.

3. The **ContextRouter** fetches all embeddings previously stored in the context store
   for this `workflow_id`. On run 1 the corpus is empty (cold start). On runs 2–4 it
   contains the embeddings of all tasks whose LLM responses were computed in prior runs.

4. Cosine similarity is computed between the query and every corpus embedding:

   ```
   similarity(q, c) = (q · c) / (‖q‖ · ‖c‖)
   ```

5. If `max(similarity) ≥ threshold` (default **0.85**), the prior LLM response is
   returned directly. Otherwise the LLM is called and the new triple
   (task_description, embedding, response) is stored for future reuse.

**What the embeddings represent:**
The embeddings encode the **semantic meaning of the task description text**, not numerical
parameters alone. Two epidemiology tasks with R0=1.8 and R0=1.9 produce embeddings with
similarity ≈ 0.92 (nearly identical scientific language). A vaccination task and a basic
SIR model task produce similarity ≈ 0.60 (semantically distinct despite same domain).

**SIMD acceleration backends:**

| Runtime | Backend | Implementation |
|---------|---------|----------------|
| Python (Lambda) | numpy | `np.dot(q, C.T)` batch matrix-vector multiply |
| Python (pycylon) | C++ SIMD | Apache Arrow SIMD float32 dot product |
| Node.js | WASM SIMD128 | cylon-wasm with `target_feature = "+simd128"` |
| GPU (ECS) | cuPy / gcylon | CUDA batch cosine similarity |

---

## 4. Experiment Variables

| Variable | Type | Values |
|----------|------|--------|
| World size (parallel workers) | Independent | 1, 2, 4, 8, 16, 32, 64 |
| Scaling mode | Independent | Weak (N × 16 tasks), Strong (64 tasks ÷ N) |
| Scientific domain | Independent | Epidemiology, Hydrology, Seismology, Mixed Scientific |
| Runtime | Independent | Python (pycylon), Node.js (WASM + cylon-node) |
| Context backend | Independent | Redis (Phase 1), FMI/TCPunch direct (Phase 2) |
| Cache hit rate | Dependent | 0% – 100% |
| Average task latency | Dependent | 50ms – 6500ms |
| Wall-clock time | Dependent | varies with world size and reuse |
| Cost savings | Dependent | 0% – 97% |

---

## 5. Example Payloads — End-to-End Data Flow

The following shows exact data structures at each pipeline stage for a 2-worker
epidemiology experiment.

### Step 1 — Step Functions trigger (cloud_sweep.py → armada_init)

```json
{
  "workflow_id": "sweep-lambda_python_epidemiology_ws2_run1-a3f1c2b4",
  "world_size": 2,
  "scaling": "weak",
  "experiment_name": "lambda_python_epidemiology_weak_ws2_run1",
  "context_backend": "redis",
  "results_s3_dir": "results/lambda-python/epidemiology/weak/",
  "tasks": [
    "Model the spread of an influenza-like illness with R0=1.8, mean generation time 3.2 days, in a metropolitan area of 2.1 million. Estimate peak timing and attack rate under no intervention.",
    "Analyze the epidemic trajectory for a respiratory pathogen with R0=1.9, serial interval 3.0 days, in an urban population of 2.0 million. Predict the infection peak and final size."
  ],
  "config": {
    "llm_model_id": "amazon.nova-lite-v1:0",
    "embedding_model_id": "amazon.titan-embed-text-v2:0",
    "embedding_dimensions": 1024,
    "similarity_threshold": 0.85,
    "region": "us-east-1"
  }
}
```

### Step 2 — armada_init → Redis + Step Functions state

`armada_init` embeds both tasks via Bedrock, stores each embedding in Redis, and returns
a **minimal** body array (no embedding bytes in Step Functions state — avoids 256KB limit):

Redis keys written:
- `task:sweep-...-a3f1c2b4:0` → `"Model the spread of an influenza-like illness..."`
- `task:sweep-...-a3f1c2b4:1` → `"Analyze the epidemic trajectory..."`
- `embedding:sweep-...-a3f1c2b4:0` → 4096 bytes (1024 × float32, base64)
- `embedding:sweep-...-a3f1c2b4:1` → 4096 bytes

Step Functions state returned to Map:
```json
{
  "workflow_id": "sweep-lambda_python_epidemiology_ws2_run1-a3f1c2b4",
  "world_size": 2,
  "body": [
    {
      "rank": 0,
      "embedding_key": "embedding:sweep-...-a3f1c2b4:0",
      "embedding_metadata": { "model_id": "amazon.titan-embed-text-v2:0",
                               "dimensions": 1024, "token_count": 38, "latency_ms": 312.4 }
    },
    {
      "rank": 1,
      "embedding_key": "embedding:sweep-...-a3f1c2b4:1",
      "embedding_metadata": { "model_id": "amazon.titan-embed-text-v2:0",
                               "dimensions": 1024, "token_count": 35, "latency_ms": 298.1 }
    }
  ]
}
```

### Step 3 — armada_executor receives per-rank payload (Map state)

```json
{
  "rank": 0,
  "world_size": 2,
  "workflow_id": "sweep-lambda_python_epidemiology_ws2_run1-a3f1c2b4",
  "embedding_key": "embedding:sweep-...-a3f1c2b4:0",
  "embedding_metadata": { "dimensions": 1024, "token_count": 38 },
  "scaling": "weak",
  "context_backend": "redis",
  "fmi_channel_type": "direct",
  "experiment_name": "lambda_python_epidemiology_weak_ws2_run1",
  "config": {
    "llm_model_id": "amazon.nova-lite-v1:0",
    "embedding_model_id": "amazon.titan-embed-text-v2:0",
    "embedding_dimensions": 1024,
    "similarity_threshold": 0.85,
    "region": "us-east-1"
  }
}
```

The executor fetches `task:{wf}:{rank}` and `embedding:{wf}:{rank}` from Redis,
then searches the context store for similar embeddings.

### Step 4 — armada_executor result (cache miss, run 1)

Run 1: context store is empty, every task hits the LLM:

```json
{
  "rank": 0,
  "source": "llm",
  "response": "To model the spread of an influenza-like illness with R0=1.8 ...",
  "context_id": "78fa5b26-8116-4d2a-b9b1-0297c4e9ab46",
  "similarity": 0.0,
  "cost_usd": 0.050355,
  "avoided_cost_usd": 0.0,
  "llm_latency_ms": 3246.94,
  "search_latency_ms": 1.2,
  "total_latency_ms": 3612.5
}
```

The executor stores `result:{experiment_name}:{rank}` in Redis and returns only
`{"rank": 0}` to Step Functions (keeps SFN state minimal).

### Step 5 — armada_executor result (cache hit, run 2)

Run 2 reuses `workflow_id` from run 1. The context store contains run 1's embedding.
Cosine similarity = 0.923 ≥ 0.85 threshold → cache hit:

```json
{
  "rank": 0,
  "source": "cache",
  "response": "To model the spread of an influenza-like illness with R0=1.8 ...",
  "context_id": "78fa5b26-8116-4d2a-b9b1-0297c4e9ab46",
  "similarity": 0.923,
  "cost_usd": 0.0,
  "avoided_cost_usd": 0.050355,
  "llm_latency_ms": 0,
  "search_latency_ms": 1.8,
  "total_latency_ms": 54.3
}
```

Latency drops from 3612ms → 54ms. No Bedrock call made.

### Step 6 — armada_aggregate output (_metrics.json written to S3)

```json
{
  "experiment_name": "lambda_python_epidemiology_weak_ws2_run2",
  "workflow_id": "sweep-lambda_python_epidemiology_ws2_run1-a3f1c2b4",
  "platform": "lambda",
  "scaling": "weak",
  "world_size": 2,
  "task_count": 32,
  "cache_hits": 28,
  "llm_calls": 4,
  "reuse_rate": 0.875,
  "total_cost": 0.006,
  "baseline_cost": 0.048,
  "savings_pct": 87.5,
  "avg_latency_ms": 312.4,
  "wall_clock_ms": 1840.2,
  "p50_latency_ms": 54.3,
  "p95_latency_ms": 3612.5
}
```

---

## 6. Cosmic AI — AstroMAE Integration

### What Cosmic AI Is

Cosmic AI (cylon-armada `cosmic_ai` module) connects the **AstroMAE deep learning model**
(arXiv:2501.06249 — *Scalable Cosmic AI Inference using Cloud Serverless Computing with FMI*)
to the cylon-armada context reuse framework.

AstroMAE is a pre-trained autoencoder that predicts **photometric redshifts** of galaxies
from multi-band SDSS (Sloan Digital Sky Survey) images and magnitude measurements.
Photometric redshift (photo-z) estimation is a core astronomy workload: spectroscopic
redshifts are expensive to obtain, so the survey community uses deep learning to predict
redshift from photometry at scale.

### What Is Being Measured in Cosmic AI Experiments

The Cosmic AI experiments measure the same four quantities as the other domains, but
with a two-stage pipeline:

**Stage 1 — ONNX Inference (AstroMAE)**

The AstroMAE model is exported to ONNX and run on Lambda (Node.js via ONNX Runtime,
Python via PyTorch/ONNX). It takes as input:

```
images:     Float32 tensor  [B, 5, 224, 224]   ← B galaxies × 5 SDSS bands × 224×224 pixels
magnitudes: Float32 tensor  [B, 5]              ← u, g, r, i, z band magnitudes
```

And produces:

```
predictions: Float32 tensor [B, 1]   ← predicted photometric redshift z_pred for each galaxy
```

Metrics recorded per batch:
- `mae` — Mean Absolute Error |z_pred − z_true|
- `mse` — Mean Squared Error
- `bias` — mean Δz/(1+z) (systematic offset)
- `precision_nmad` — 1.48 × median(|Δz/1+z − median(Δz/1+z)|) (robust scatter)
- `r2` — R² coefficient of determination
- `total_time_s`, `throughput_bps`, `samples_per_sec` — inference performance

**Stage 2 — LLM Task Generation and Context Reuse**

The inference results (predictions, true redshifts, magnitudes) are fed into
`task_generator.py`, which generates **semantically clustered LLM analysis tasks**:

| Template | Triggered when | Example |
|----------|---------------|---------|
| `redshift_analysis` | Normal prediction (even index) | Analyze z_pred vs z_true for galaxy with band magnitudes |
| `color_classification` | Normal prediction (odd index) | Classify morphology from color indices |
| `outlier_analysis` | Residual in top 10% | Explain why AstroMAE predicted z=0.62 when true z=0.31 |
| `batch_summary` | Per batch (2 per run) | Summarize batch MAE=0.021, bias, NMAD for survey suitability |
| `cost_analysis` | Per inference run | Compare serverless vs HPC cost for N galaxies at measured throughput |

### Cosmic AI Example Payload

**armada_init input** (generated from AstroMAE inference results):

```json
{
  "workflow_id": "cosmic-ai-ws4-run1-b7e2f9a1",
  "world_size": 4,
  "scaling": "weak",
  "experiment_name": "lambda_python_cosmic_ws4_run1",
  "context_backend": "redis",
  "tasks": [
    "Analyze the photometric redshift prediction z=0.451 (true z=0.443) for a galaxy with SDSS magnitudes u=22.13, g=20.91, r=20.34, i=19.74, z=19.12. Assess the prediction accuracy and classify the likely galaxy morphological type based on the color profile.",
    "Given SDSS color indices u-g=1.22, g-r=0.57, r-i=0.60, i-z=0.62 and predicted redshift z=0.389, classify this galaxy's morphological type and assess whether the colors are consistent with the predicted redshift.",
    "The AstroMAE model predicted z=0.621 for a galaxy with true spectroscopic redshift z=0.312 (residual=0.3090). The galaxy has magnitudes u=23.41, g=22.18, r=21.52, i=20.91, z=20.44. Analyze whether this prediction error is significant and identify possible causes.",
    "Summarize the inference results for a batch of 512 galaxies: mean predicted redshift z=0.387, MAE=0.0214, bias=-0.0031, precision(NMAD)=0.0189. Assess whether this accuracy meets the requirements for large-scale structure surveys."
  ],
  "config": {
    "llm_model_id": "amazon.nova-lite-v1:0",
    "embedding_model_id": "amazon.titan-embed-text-v2:0",
    "embedding_dimensions": 1024,
    "similarity_threshold": 0.85,
    "region": "us-east-1"
  }
}
```

**What the context reuse captures in Cosmic AI:**
- Two `redshift_analysis` tasks for galaxies at z≈0.45 with similar magnitudes → similarity ≈ 0.91 → reuse
- A `batch_summary` for MAE=0.021 vs another for MAE=0.022 → similarity ≈ 0.94 → reuse
- A `color_classification` for an elliptical vs a `redshift_analysis` for a spiral → similarity ≈ 0.58 → LLM call

### What Is Measured (Cosmic AI specific)

In addition to the standard cost/latency metrics, Cosmic AI experiments track:

| Metric | What it measures |
|--------|-----------------|
| `mae` | AstroMAE prediction accuracy (photo-z vs spec-z) |
| `precision_nmad` | Robust scatter of redshift predictions |
| `throughput_bps` | ONNX inference throughput (Gbps) |
| `samples_per_sec` | Galaxy inference rate |
| `model_parallelism_overhead_ms` | Extra latency when splitting AstroMAE across 2 Lambda workers via FMI (ViT encoder on rank 0, Inception on rank 1, tensor exchange via FMI) |

---

## 7. Complete Scientific Query Catalogue

### 7.1 Epidemiology (32 tasks)

Tasks model infectious disease dynamics (SIR/SEIR, vaccination, interventions).
Parameter variation (R0, coverage %, population size) produces controlled semantic similarity.

1. Model the spread of an influenza-like illness with R0=1.8, mean generation time 3.2 days, in a metropolitan area of 2.1 million. Estimate peak timing and attack rate under no intervention.
2. Analyze the epidemic trajectory for a respiratory pathogen with R0=1.9, serial interval 3.0 days, in an urban population of 2.0 million. Predict the infection peak and final size.
3. Evaluate the effectiveness of a vaccination campaign achieving 60% coverage with a vaccine of 85% efficacy against a pathogen with R0=2.5. What is the herd immunity threshold?
4. Assess the impact of vaccinating 55% of the population with a 90% effective vaccine against a disease with R0=2.3. Does coverage achieve herd immunity?
5. Model the impact of school closures on disease transmission in a community with 18% school-age population. The pathogen has R0=1.6 with 40% of transmission occurring in schools.
6. Analyze how closing schools (affecting 20% of the population) reduces transmission of a respiratory illness with R0=1.7, assuming 35% of contacts occur in educational settings.
7. Estimate the contact tracing capacity needed to contain an outbreak with 50 initial cases, R0=2.0, and a 4-day serial interval. What fraction of contacts must be traced within 48 hours?
8. Assess the feasibility of contact tracing for an emerging pathogen with 65 initial cases, R0=2.2, and 3.5-day generation time. How many contact tracers per case are required?
9. Model the spatial spread of a vector-borne disease with Aedes aegypti density of 12 per household, average flight range 200m, and extrinsic incubation period of 10 days in a tropical city of 500K.
10. Analyze the geographic diffusion pattern of a mosquito-borne illness in a city of 480K, with vector density 14 per household, 180m dispersal range, and 12-day extrinsic incubation.
11. Evaluate the cost-effectiveness of three non-pharmaceutical interventions for COVID-19: universal masking (50% reduction), social distancing (30% reduction), and workplace closures (25% reduction).
12. Compare the epidemiological impact of mask mandates, capacity limits, and remote work policies for a respiratory pathogen in a county of 350K residents.
13. Model the emergence of antiviral resistance in an influenza epidemic treated with oseltamivir. Initial resistance frequency is 0.1%, with a fitness cost of 5% and treatment coverage of 30%.
14. Analyze the evolutionary dynamics of drug resistance during an influenza outbreak where 25% of cases receive antivirals. Resistance mutation confers 3% fitness cost but full treatment escape.
15. Assess the impact of population age structure on disease severity for a pathogen with age-dependent IFR: 0.01% (0-19), 0.1% (20-49), 1% (50-69), 5% (70+). Compare young vs aging populations.
16. Analyze how demographic differences (median age 28 vs 42) affect hospitalization burden for a respiratory pathogen with age-stratified severity rates.
17. Model superspreading dynamics where 10% of infected individuals cause 80% of secondary infections. How does this overdispersion (k=0.1) affect outbreak probability and control strategies?
18. Analyze the role of superspreading events (dispersion parameter k=0.15) in driving epidemic dynamics. What is the probability of stochastic extinction vs sustained transmission?
19. Evaluate a syndromic surveillance system using emergency department chief complaint data for early detection of respiratory outbreaks. What is the expected detection delay at 95% sensitivity?
20. Assess the performance of wastewater-based epidemiology for tracking SARS-CoV-2 prevalence. How does sewershed population size affect the detection limit and lead time over clinical surveillance?
21. Model the impact of seasonal forcing on endemic equilibrium for a pathogen with R0=3.0 and 20% seasonal variation in transmission. Predict the amplitude and timing of annual epidemic waves.
22. Analyze how climate-driven seasonality (15% transmission variation) affects the inter-epidemic period for an endemic respiratory pathogen with R0=2.8.
23. Estimate the probability of importation-driven outbreaks from a region with 10,000 active cases, given 500 daily air travelers and a 1% prevalence among travelers.
24. Assess the risk of disease introduction from an endemic region via air travel: 450 daily passengers, 1.2% infection prevalence, 70% asymptomatic. Model border screening effectiveness.
25. Model a two-strain pathogen system where strain B has 30% higher transmissibility but cross-immunity of 60% from prior strain A infection. Predict strain replacement dynamics.
26. Analyze competitive dynamics between an ancestral strain (R0=2.5) and an immune-evasive variant (R0=3.0, 40% immune escape) in a population with 50% prior infection.
27. Evaluate optimal allocation of a limited vaccine supply (covering 20% of the population) across age groups to minimize deaths vs minimize transmission for a pathogen with age-dependent severity.
28. Determine the optimal vaccine distribution strategy for 15% population coverage: prioritize healthcare workers, elderly, or essential workers?
29. Model the impact of behavioral changes (voluntary social distancing) on epidemic dynamics. Assume individuals reduce contacts by 40% when local prevalence exceeds 2%.
30. Analyze how prevalence-dependent behavior modification creates feedback loops that flatten epidemic curves without mandated interventions.
31. Assess the reliability of case fatality rate estimates during the early phase of an epidemic with 500 confirmed cases, 15 deaths, and an estimated reporting rate of 30%.
32. Evaluate the biases in real-time severity estimation for an emerging pathogen with 800 detected cases, 20 deaths, right-censored outcomes, and preferential testing of severe cases.

---

### 7.2 Hydrology (32 tasks)

Tasks model watershed hydrology (flood risk, drought, water quality, climate impacts).

1. Analyze the flood risk for a watershed with drainage area 245 km², mean annual precipitation 1120 mm, land cover 65% forested, and soil type predominantly clay loam. Estimate the 100-year return period peak discharge.
2. Assess the flood hazard for a catchment of 238 km² receiving 1090 mm annual rainfall, with 62% forest cover and clay loam soils. Calculate the expected peak flow for a 100-year event.
3. Evaluate the hydrological response of a 312 km² watershed with 890 mm annual precipitation, 45% agricultural land use, and sandy loam soils. How does the land use affect runoff coefficient?
4. Analyze the runoff characteristics of a 298 km² catchment receiving 920 mm annual rainfall, with 48% cropland and sandy loam soil. Estimate the curve number and peak discharge.
5. Assess the streamflow regime for a mountainous watershed at elevation 1800-3200m, area 156 km², snowmelt-dominated with 1450 mm annual precipitation. Predict the timing and magnitude of spring peak flows.
6. Analyze the snowmelt hydrology of a high-elevation catchment (1750-3100m, 162 km²) receiving 1480 mm annual precipitation. How does the snow water equivalent affect the hydrograph?
7. Evaluate the drought vulnerability of a 520 km² semi-arid basin with 380 mm annual precipitation, 78% rangeland, and groundwater-dependent baseflow. What is the probability of consecutive dry years?
8. Assess the water stress for a 485 km² arid watershed receiving 410 mm annual rainfall, with 75% grassland and shallow groundwater table. Analyze the baseflow recession characteristics.
9. Analyze the water quality impacts of urbanization in a 89 km² watershed where impervious surface increased from 12% to 35% over 20 years.
10. Evaluate how urban expansion (impervious cover rising from 15% to 38%) in a 92 km² catchment affects stormwater quality, focusing on phosphorus and suspended sediment concentrations.
11. Assess the effectiveness of riparian buffer strips (30m width) for reducing nitrogen loading in an agricultural watershed with tile drainage and 180 kg/ha fertilizer application.
12. Analyze the nutrient removal efficiency of a 25m riparian zone in a catchment with intensive agriculture and subsurface drainage. What percentage of nitrate load is attenuated?
13. Model the impact of a 2°C temperature increase on the hydrological cycle of a 680 km² watershed in the mid-Atlantic region. How do evapotranspiration and soil moisture change?
14. Analyze climate change effects (+2.1°C warming) on water balance for a 710 km² mid-Atlantic catchment. Predict shifts in seasonal streamflow distribution.
15. Evaluate how projected precipitation increases of 8-12% affect flood frequency in a coastal plain watershed with high water table and tidal influence.
16. Assess the compound flood risk from combined riverine flooding and storm surge in a 340 km² coastal watershed under a 10% precipitation increase scenario.
17. Analyze the groundwater recharge rate for an alluvial aquifer system with hydraulic conductivity 15 m/day, receiving 650 mm infiltration annually. Estimate sustainable yield.
18. Evaluate the sustainable pumping rate for a confined aquifer with transmissivity 800 m²/day, storativity 0.0003, and 12 production wells.
19. Model the sediment transport dynamics in a 420 km² watershed with steep terrain (mean slope 18°), 1200 mm annual rainfall, and recent timber harvest on 15% of the area.
20. Analyze erosion and sediment yield for a 395 km² mountainous catchment with 20° mean slope, 1150 mm precipitation, and active logging on 12% of the watershed.
21. Assess the effectiveness of three flood mitigation strategies for a 200 km² urban watershed: upstream detention basins, channel widening, and floodplain restoration.
22. Compare flood reduction benefits of green infrastructure (bioswales, rain gardens) versus grey infrastructure (stormwater pipes, detention) for a 185 km² suburban catchment.
23. Analyze the impact of dam removal on downstream sediment dynamics, fish passage, and flood characteristics for a 45m high dam on a 3rd-order stream.
24. Evaluate the ecological and hydrological consequences of removing a 40m dam from a regulated river system, including sediment pulse modeling and habitat restoration potential.
25. Model the hydrological connectivity between surface water and groundwater in a wetland complex spanning 28 km² with seasonal water table fluctuations of 0.8-2.1m.
26. Analyze the ecohydrology of a 32 km² palustrine wetland system with groundwater-surface water exchange. How do seasonal water level changes affect vegetation communities?
27. Assess the water budget for an irrigated agricultural district consuming 180 million m³/year from a river system with 450 million m³/year mean annual flow.
28. Analyze the water allocation sustainability for an irrigation scheme using 175 million m³/year from a river with 460 million m³/year average discharge.
29. Evaluate the flood forecasting accuracy for a 1200 km² river basin using a distributed hydrological model with 6-hour rainfall forecasts. What lead time is achievable?
30. Assess the predictive skill of real-time flood forecasting in a 1150 km² watershed using radar rainfall and a semi-distributed model. Quantify forecast uncertainty at 12-hour lead time.
31. Model the impact of land use change (conversion of 20% wetland to agriculture) on peak flows, baseflow, and water quality in a 350 km² lowland watershed.
32. Analyze how wetland loss (18% converted to cropland) affects the hydrological regime of a 380 km² floodplain catchment, focusing on flood attenuation and nutrient retention.

---

### 7.3 Seismology (32 tasks)

Tasks cover probabilistic seismic hazard, ground motion prediction, and earthquake engineering.

1. Assess the seismic hazard for a site 15 km from an active strike-slip fault with a slip rate of 8 mm/yr and Mmax=7.2. Estimate the 475-year return period PGA.
2. Evaluate the earthquake hazard at a location 18 km from a strike-slip fault slipping at 7.5 mm/yr with Mmax=7.0. Calculate probabilistic ground motion for 10% exceedance in 50 years.
3. Analyze the ground motion prediction for a M6.5 earthquake at 25 km epicentral distance on a VS30=360 m/s site. Compare NGA-West2 GMPE estimates for PGA and SA at 0.2s and 1.0s.
4. Estimate ground shaking from a M6.3 event at 28 km distance on a site with VS30=380 m/s using multiple ground motion models. Quantify epistemic uncertainty in spectral acceleration.
5. Characterize the seismogenic potential of a 120 km thrust fault segment with 3 mm/yr slip rate and paleoseismic evidence of 4 events in the last 8000 years. Estimate recurrence interval.
6. Analyze the earthquake recurrence for a 135 km reverse fault with 2.8 mm/yr slip rate. Paleoseismic trenching reveals 5 surface-rupturing events over 10,000 years. Fit a renewal model.
7. Model the aftershock sequence following a M7.0 mainshock using the modified Omori-Utsu law. Predict the number of M≥4.0 aftershocks in the first 30 days.
8. Forecast aftershock activity for a M6.8 mainshock using ETAS parameters. Estimate the probability of a M≥5.5 event within 7 days.
9. Evaluate the liquefaction susceptibility for a site with SPT blow counts N=12, groundwater depth 2m, and an M7.0 design earthquake at 20 km. Use the Boulanger-Idriss method.
10. Assess liquefaction potential at a coastal site with N-values averaging 14, water table at 1.8m depth, subject to a M6.8 scenario earthquake at 22 km. Calculate factor of safety.
11. Analyze the seismic performance of a 12-story RC frame building designed to 1990s code standards subjected to a M7.0 near-field ground motion with PGA=0.45g.
12. Evaluate the collapse probability of a 10-story RC moment frame (pre-2000 design) under a M6.8 earthquake producing PGA=0.40g at the building site. Apply FEMA P-58 methodology.
13. Model the cascading effects of a M7.5 earthquake on a regional lifeline network: water distribution, power grid, and transportation. Estimate system-level restoration time.
14. Analyze infrastructure interdependency failures following a M7.2 earthquake: power outage impacts on water pumping stations, hospital operations, and communication networks.
15. Assess the tsunami hazard for a coastal city from a M8.5 subduction zone earthquake 150 km offshore. Estimate wave arrival time, maximum run-up height, and inundation extent.
16. Model the tsunami generated by a M8.3 megathrust event 180 km from shore. Calculate propagation time, coastal amplification, and vulnerable evacuation zones.
17. Evaluate the seismic risk for a portfolio of 500 buildings in a moderate-hazard region (PGA10%/50yr = 0.25g) with a mix of construction types. Estimate annual expected loss.
18. Assess the earthquake insurance exposure for a building portfolio of 480 structures across 3 seismic zones. Calculate probable maximum loss at the 250-year return period.
19. Analyze the effectiveness of earthquake early warning for a M6.5 event detected 80 km from a major city. Estimate available warning time and achievable automated protective actions.
20. Evaluate ShakeAlert performance for a M6.2 earthquake 90 km from the target city. Calculate P-wave travel time, alert latency, and useful warning seconds.
21. Model induced seismicity from wastewater injection at 15,000 barrels/day into a formation 3 km deep, 5 km from a mapped fault. Assess the probability of M≥3.0 events.
22. Analyze the seismic hazard from fluid injection operations disposing 12,000 bbl/day at 2.8 km depth near a critically stressed fault. Apply the McGarr maximum magnitude relationship.
23. Characterize the b-value and completeness magnitude for a seismic catalog of 15,000 events recorded over 20 years in a tectonically active region.
24. Analyze the frequency-magnitude distribution for a regional earthquake catalog (12,000 events, 25 years). Estimate Gutenberg-Richter parameters and detection threshold.
25. Evaluate site amplification effects for a deep sedimentary basin (depth 3 km) with VS30=250 m/s. Model basin-edge amplification and resonance frequencies.
26. Analyze the site response for a location on a 2.5 km deep alluvial basin with VS30=280 m/s. How do basin geometry and impedance contrasts affect long-period ground motions?
27. Assess the seismic vulnerability of unreinforced masonry structures (pre-1940) in a moderate-hazard zone. Develop fragility curves for slight, moderate, and complete damage states.
28. Generate damage fragility functions for a class of pre-code URM buildings subjected to ground motions with PGA 0.1-0.6g. Compare analytical and empirical approaches.
29. Model the stress transfer and Coulomb failure stress changes on nearby faults following a M7.0 mainshock on a N30W striking, 60° dipping reverse fault.
30. Analyze the static stress triggering potential of a M6.8 earthquake on adjacent fault segments. Calculate Coulomb stress changes and identify fault segments brought closer to failure.
31. Evaluate a machine learning approach for earthquake magnitude prediction using 50,000 waveform features. Compare random forest, gradient boosting, and neural network performance.
32. Assess the predictive skill of a deep learning model trained on 45,000 seismic waveforms for magnitude estimation. Analyze feature importance and generalization across tectonic settings.

---

### 7.4 Mixed Scientific / Cosmic AI (48 tasks)

This domain combines the three scientific domains above with Cosmic AI
(photometric redshift) tasks and framework self-benchmarking tasks.

**Astronomy / Cosmic AI (12 tasks):**
1. Analyze the photometric redshift prediction z=0.45 for an elliptical galaxy with SDSS magnitudes u=22.1, g=20.9, r=20.3, i=19.7, z=19.1. Classify the galaxy type.
2. Evaluate the redshift estimate z=0.43 for a galaxy with magnitudes u=22.3, g=21.0, r=20.4, i=19.8, z=19.2. Is this consistent with an elliptical morphology?
3. Assess the AstroMAE model accuracy for a batch of 256 galaxies with mean predicted redshift z=0.31, MAE=0.021. How does this compare to spectroscopic requirements?
4. Analyze the cost-effectiveness of serverless inference for 500K galaxy images at $0.38 per 12.6GB partition versus HPC GPU at $2.50/hour.
5. For a spiral galaxy with color indices u-g=0.89, g-r=0.58, r-i=0.55, i-z=0.32, classify the morphological type and estimate the star formation rate.
6. Evaluate the photometric redshift bias for a sample of 1000 galaxies at z>1.0. What systematic effects cause catastrophic outliers?

**Cross-domain hydrology (6 tasks):**
7–12. Flood risk, climate change, groundwater, and sediment transport queries (variants of the hydrology domain above, included for cross-domain similarity testing).

**Cross-domain epidemiology (6 tasks):**
13–18. SIR model, vaccination, school closure, contact tracing, superspreading queries.

**Cross-domain seismology (6 tasks):**
19–24. Seismic hazard, ground motion, aftershock, liquefaction, and tsunami queries.

**Framework benchmarking (18 tasks):**
25–48. Tasks that directly measure the framework's own performance — these are unique
to cylon-armada and test the system under study rather than the scientific domains:

- Compare SIMD-accelerated similarity search versus brute-force numpy for 1000 embeddings of dimension 1024.
- Analyze the latency distribution for Cylon's cosine_similarity_f32 across batch sizes of 100, 500, 1000, and 5000 embeddings at 256 dimensions.
- Evaluate the cost savings from context reuse at similarity thresholds 0.70, 0.80, and 0.90 for a 32-task workflow.
- Benchmark the three execution paths (A1: pycylon per-call, A2: Cython batch, B: WASM SIMD128) for 1000 cosine similarity computations at 512 dimensions.
- Analyze the embedding dimension trade-off: 256 vs 512 vs 1024 dimensions for Amazon Titan V2. How does dimensionality affect reuse accuracy and search latency?
- Evaluate the Arrow IPC serialization overhead for context tables of 100, 1000, and 10000 entries with 1024-dimensional embeddings.
- Model the FMI communication overhead for broadcasting a context table with 500 embeddings (1024-dim) from rank 0 to 10 Lambda workers via Redis channel.
- Compare the total workflow cost for Step Functions orchestration versus direct Lambda invocation via Cylon FMI communicator for a 16-task workflow.
- Analyze the cold start latency for Python Lambda (pycylon) versus Node.js Lambda (cylon-wasm + ONNX Runtime). What dominates initialization time?
- Evaluate the model parallelism overhead: single Lambda full inference versus 2-Lambda split (ViT + Inception) with FMI tensor exchange for AstroMAE.
- Assess the memory efficiency of Cylon ContextTable versus raw numpy+Redis for 5000 contexts with 1024-dimensional embeddings.
- Analyze the cache hit rate as a function of task ordering: sequential similar tasks versus randomized task order for a 32-task workflow.
- Assess the cross-domain reuse rate: do hydrology tasks ever reuse epidemiology contexts? Measure false positive rates.
- Evaluate the quality of reused responses using ROUGE-L and BERTScore for near-threshold matches (similarity 0.83–0.87).
- Model the scalability of the context store: how does similarity search latency grow from 100 to 100,000 entries?
- Analyze the pricing accuracy of BedrockCostTracker versus actual AWS billing for a 48-task experiment.
- Benchmark ONNX Runtime inference latency for AstroMAE on Lambda (Node.js) versus PyTorch on Lambda (Python) for batches of 32 and 128 images.
- Evaluate the consistency between Python and Node.js cost tracking: run the same 16-task workflow on both paths and compare cost summaries.

---

## 8. Actual Measured Results (Phase 1, Lambda, Completed)

### Epidemiology — Weak Scaling, Python Lambda (average over 4 runs per world size)

| World Size | Runs | Avg Reuse Rate | Avg Task Latency |
|:----------:|:----:|:--------------:|:----------------:|
| 1          | 32   | 0% (run 1 cold) → high (runs 2–4) | ~3650ms |
| 2          | 24   | 23.4%          | 2812ms           |
| 4          | 16   | 32.1%          | 2471ms           |
| 8          | 16   | 36.3%          | 2200ms           |
| 16         | 12   | 45.9%          | 1852ms           |
| 32         | 8    | 46.6%          | 1454ms           |
| 64         | 8    | 48.4%          | 737ms            |

**Key observation**: Reuse rate increases with world size (more workers, more context overlap).
Average latency decreases as reuse saturates (cached response ~50ms vs cold LLM call ~3500ms).

Cost: run 1 at ws=1 costs ~$0.030. At ws=64 run 4, savings approach **97%** across domains.

---

## 9. What Is Not Being Measured

- **LLM response quality** — tasks are routed, not evaluated for correctness.
  The research question is about *cost and latency of routing*, not answer quality.
- **Network bandwidth** between Lambda workers — FMI experiments measure latency
  improvement from eliminating Redis round-trips, not raw throughput.
- **Absolute HPC performance** (FLOPS, memory bandwidth) — this is an inference
  routing framework, not a numerical simulation framework.

---

## 10. Relationship to Prior Work

The cylon-armada experiments extend the Cylon distributed DataFrame framework
(`/home/parallels/cylon/target/shared/scripts/scaling/scaling.py`) which measured:
- Join, GroupBy, AllReduce throughput on FMI/UCC/UCX backends
- Weak and strong scaling of data-parallel operations on Lambda and ECS

Cylon Armada applies the same FMI collective communication infrastructure (barrier,
broadcast, allreduce) to a new problem: **LLM context sharing** rather than DataFrame
operations. TCPunch TCP hole-punching that enables direct P2P connections between Lambda
functions behind NAT is the same mechanism used in the Cylon scaling experiments.

The AstroMAE model is from arXiv:2501.06249 — *Scalable Cosmic AI Inference using Cloud
Serverless Computing with FMI*, which established the serverless astronomy inference
baseline that cylon-armada now extends with context reuse.

---

*Last updated: May 2026*
*Phase 1: 464 runs completed (Lambda Python + Node.js, ws1–ws64, 4 domains)*
*Phase 2 (FMI direct TCPunch): in progress*