
# 📦 Delivery Time Prediction

A modular machine learning pipeline that predicts food delivery times from order, weather, and traffic
data. Built with a clean component-based architecture and served through an interactive **Streamlit** app.

---

## 🚀 Key Features

- 🔄 **Modular pipeline** — ingestion, transformation, and training are independent, testable components
- 📥 **Data ingestion** with schema validation against `schema.yml`
- 🧹 **Feature engineering & preprocessing** (distance calculation, time parsing, encoding, scaling)
- 🎯 **Model selection** across five regressors, picked automatically by R² score
- 📈 **Batch prediction** CLI for scoring a CSV of records at once
- 🌐 **Streamlit app** for interactive, single-record predictions
- ⚙️ **Config-driven** — split ratios, random seeds live in `config/config.yml`, not hardcoded
- ✅ Unit tests for the exception, logging, and pipeline utilities

---

## 🗂️ Project Structure

```
moduler-project/
│
├── app.py                     # Streamlit app entry point
├── main.py                    # Runs the training pipeline end-to-end
├── pyproject.toml             # Packaging + dependencies (single source of truth, managed by uv)
├── uv.lock                    # Locked, reproducible dependency versions
├── schema.yml                 # Expected raw-data columns, checked at ingestion time
│
├── config/
│   └── config.yml             # Split ratios, random seeds
│
├── data/
│   └── finalTrain.csv         # Source dataset
│
├── batch_prediction/
│   └── predict.py             # CLI: score a CSV of records with the trained model
│
├── scripts/
│   └── generate_project_structure.py   # Scaffolds a new project with this same layout
│
├── tests/                     # pytest unit tests
│
├── artifacts/                 # Generated at runtime (models, transformed data) — gitignored
│
└── src/
    ├── components/            # One file per pipeline stage
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    ├── config/
    │   └── configuration.py   # Resolves every artifact path used by the pipeline
    ├── constants/              # Directory names, file names, default hyperparameters
    ├── entity/                 # Data-class definitions shared across components
    ├── exceptions/             # CustomException — wraps errors with file/line context
    ├── logger/                 # Timestamped file logging setup
    ├── pipeline/
    │   ├── training_pipeline.py
    │   └── prediction_pipeline.py
    └── utils/                  # save/load model, YAML loading, schema validation, R² scoring
```

---

## 🏷️ Naming Convention

This project follows [PEP 8](https://peps.python.org/pep-0008/):

| Element                     | Convention           | Example                          |
|------------------------------|-----------------------|-----------------------------------|
| Packages / modules / folders | `snake_case`          | `src/components`, `data_ingestion.py` |
| Classes                       | `PascalCase`          | `DataIngestion`, `CustomException` |
| Functions / variables         | `snake_case`          | `initiate_data_ingestion`, `train_data_path` |
| Module-level constants        | `UPPER_SNAKE_CASE`    | `TARGET_COLUMN`, `RAW_DATA_PATH` |

Two intentional exceptions: `CustomData`'s constructor arguments and the feature/config lists inside
`data_transformation.py` keep the raw dataset's original column spelling (e.g. `Time_Orderd`,
`Delivery_person_ID`) so the code stays a direct, greppable match for the CSV headers it reads.

---

## 🛠️ How to Run

Dependencies and the virtual environment are managed with [uv](https://docs.astral.sh/uv/); everything
in `pyproject.toml` is the single source of truth (no separate `requirements.txt`).

### 🔧 Install dependencies

```bash
uv sync --group dev
```

### 🚦 Train the model

Runs ingestion → transformation → training, and writes artifacts under `artifacts/`:

```bash
uv run python main.py
```

### 🧪 Run batch predictions

Scores a CSV (defaults to the held-out test split produced by training):

```bash
uv run python batch_prediction/predict.py --input data/test_data.csv --output batch_prediction/predictions.csv
```

### 🌐 Launch the Streamlit app

Requires a trained model (run `uv run python main.py` first):

```bash
uv run streamlit run app.py
```

### ✅ Run tests

```bash
uv run pytest
```

Without uv, `pip install -e .` (runtime) or `pip install -e ".[dev]"`-style extras aren't set up here —
use `pip install .` then `pip install pytest` for an equivalent pip-only environment.

---

## 📊 Streamlit App Preview

> ![App Preview](app_preview.png)

---

## 🔍 Sample Input Features

| Feature Name          | Description                                |
|------------------------|---------------------------------------------|
| `distance`             | Distance of delivery (km)                  |
| `multiple_deliveries`  | Number of concurrent deliveries            |
| `Road_traffic_density` | Road/traffic status                        |
| `Weather_conditions`   | Weather during delivery                    |
| `Time_Orderd_hour`     | Hour the order was placed                  |
| `Type_of_order`        | Order type (Snack, Meal, Drinks, Buffet)   |

---

## 📚 Tech Stack

- Python 3.10+
- scikit-learn, XGBoost
- pandas, NumPy
- Streamlit
- pytest

---

## 📝 License

This project is licensed under the [Apache 2.0 License](LICENSE).
