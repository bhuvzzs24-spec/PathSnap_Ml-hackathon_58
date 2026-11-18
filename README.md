
# PathSnap 🚗
### Intelligent GPS Map Matching Using Machine Learning and HMM-Based Route Decoding

PathSnap is a GPS map-matching system designed to convert noisy GPS observations into more meaningful road-network routes.

GPS devices often produce inaccurate or noisy location points due to signal interference, urban environments, and sampling limitations. PathSnap addresses this problem using machine learning-based road-segment prediction combined with Hidden Markov Model (HMM)-style sequence decoding.

The project includes a machine learning backend, route-decoding logic, model artifacts, experiment results, and an interactive Streamlit interface for evaluation and visualisation.

---

## ✨ Key Features

- **Machine Learning-Based Map Matching:** Uses trained classifiers to identify likely road segments for GPS observations.
- **XGBoost Model:** Includes an exported XGBoost model and its configuration.
- **Multiple Model Experiments:** Includes Logistic Regression, Random Forest, Neural Network, and SVM-related artifacts and evaluation results.
- **HMM-Based Route Decoding:** Applies sequence-level decoding to improve route continuity.
- **FastAPI Backend:** Exposes endpoints for health checks, trip listing, and route prediction.
- **Interactive Streamlit Interface:** Provides a dashboard for exploring trips and visualising predictions.
- **Folium Visualisation:** Supports geographical display of GPS observations and predicted routes.

---

## 🧠 System Architecture

```text
       GPS Trip Data
             |
             v
    Data Cleaning and
      Preprocessing
             |
             v
      Feature Extraction
             |
             v
   Machine Learning Model
       (XGBoost / Other
        Model Experiments)
             |
             v
   Candidate Road-Segment
        Predictions
             |
             v
     HMM / Viterbi-Based
       Route Decoding
             |
             v
      Predicted Route
             |
             v
       FastAPI Backend
             |
             v
    Streamlit Evaluation UI
             |
             v
      Interactive Map
       Visualisation
```

---

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Deep Learning | PyTorch |
| Data Processing | NumPy, Pandas, SciPy |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Map Visualisation | Folium, Streamlit-Folium |
| Model Storage | Joblib, XGBoost JSON, PyTorch |
| Experimentation | Jupyter Notebook |

---

## 📁 Repository Structure

```text
PathSnap/
│
├── data/
│   ├── cleaned_train/
│   └── cleaning.py
│
├── models/
│   ├── config.json
│   ├── xgboost_model.json
│   ├── logistic_map_matcher.joblib
│   ├── pathsnap_untuned_svm.joblib
│   └── neural_map_matcher_checkpoint.zip
│
├── notebooks/
│
├── results/
│   ├── hmm_comparison.csv
│   ├── logistic_student_test_results.csv
│   ├── neural_student_test_results.csv
│   ├── neural_training_history.csv
│   └── random_forest_student_test_results.csv
│
├── src/
│   ├── main.py
│   ├── features.py
│   ├── hmm.py
│   ├── json_predictor.py
│   ├── logistic.py
│   ├── neural.py
│   ├── random.py
│   ├── preprocessing.py
│   ├── evaluate.py
│   ├── evaluate_hmm.py
│   └── map_visualisation.py
│
├── ui/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/bhuvzzs24-spec/PathSnap_Ml-hackathon_58.git
cd PathSnap_Ml-hackathon_58
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the dataset

The backend expects the original dataset archive containing the trip data and road-network files.

The expected raw dataset structure includes:

```text
Student folder/
└── train/
    └── trip_id/
        ├── trip_id.track
        ├── trip_id.nodes
        ├── trip_id.arcs
        └── trip_id.route
```

Place the original dataset archive at the location expected by the backend, or configure the `PATHSNAP_DATA` environment variable with its full path.

The repository also contains cleaned GPS track files under `data/cleaned_train/`. These are not a replacement for the complete raw dataset.

---

## ▶️ Running the Application

### 1. Start the FastAPI backend

Open a terminal at the project root:

```bash
cd src
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the Streamlit interface

Open another terminal at the project root:

```bash
cd ui
streamlit run app.py
```

Open the local URL displayed by Streamlit.

Configure the backend URL in the interface if required.

---

## 📊 Experiments and Results

The `results/` directory contains the available experiment outputs:

- Logistic Regression student-test results
- Random Forest student-test results
- Neural Network student-test results
- Neural Network training history
- HMM comparison results

These files document the experiments and evaluation outputs currently included in the repository.

Additional training scripts, model artifacts, and experiment notebooks may be added as the project is consolidated.

---

## 🔬 Route Decoding

The HMM-style decoder improves route-level consistency by considering both:

- **Emission scores:** How well a candidate road segment matches the observed GPS point.
- **Transition scores:** How plausible movement between consecutive candidate road segments is.

A Viterbi-style dynamic programming approach is used to find a high-scoring sequence of road-segment candidates.

The decoder also supports disconnected transitions using a configurable penalty.

---

## 📌 Project Status and Reproducibility

This repository contains the currently assembled application code, selected trained model artifacts, cleaned GPS tracks, and available experiment outputs.

Some original experiment scripts, raw dataset files, and training artifacts are not included in the current version. Therefore, the repository should not yet be considered a fully self-contained reproduction package.

The README and repository will be updated as the remaining project files are consolidated.

---

## 👥 Contributors

Developed as part of a machine learning hackathon project.

---

## 📄 License

Add the appropriate project license before redistributing this repository.


---

## Authors
SAANVI MAHARANA | SAATHVIK SAI SUVARNA    |     S BHUVANA SHREE    |    K PREKSHA
