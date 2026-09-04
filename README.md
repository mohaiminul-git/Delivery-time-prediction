
# 📦 Delivery Time Prediction

A modular, production-style machine learning pipeline that predicts food delivery times from order,
weather, and traffic conditions — built with an independent, testable component for every pipeline
stage, config- and schema-driven behavior instead of hardcoded values, and served through an
interactive **Streamlit** app with both single-record and batch prediction modes.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![uv](https://img.shields.io/badge/deps-uv-de5fe9)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-pytest-0a9edc)](tests)
[![License](https://img.shields.io/badge/license-Apache%202.0-informational)](LICENSE)

---

## 🎥 Demo

> **[Demo recording goes here]** — a short screen recording of both the single-record sidebar flow and
> the CSV batch-upload flow. See the note at the bottom of this README for how it gets added.

https://github.com/user-attachments/assets/ad5d02b6-3fe1-48e4-a188-f8261245e259

---

## 🚀 Key Features

- 🔄 **Modular pipeline** — ingestion, transformation, and training are independent, testable components,
  each with its own config dataclass and no hidden coupling between stages
- 📥 **Data ingestion** with schema validation against `schema.yml` before anything else runs
- 🧹 **Feature engineering & preprocessing** — haversine distance from lat/long, time parsing, ordinal +
  one-hot encoding, scaling — wrapped as real scikit-learn `Pipeline`/`ColumnTransformer` steps
- 🎯 **Automatic model selection** across five regressors (XGBoost, Random Forest, Gradient Boosting,
  Decision Tree, SVR), picked by R² score on a held-out split
- 🌐 **Streamlit app** with two modes: a sidebar form for single predictions, and a CSV upload tab for
  batch scoring with a downloadable results file
- 📈 **Batch prediction CLI** (`batch_prediction/predict.py`) for scoring a CSV from the command line
- ⚙️ **Config-driven** — split ratio and random seed live in `config/config.yml`, not buried in code
- 🧾 **Structured error handling** — a `CustomException` wraps every failure with the originating file
  and line number, and every run is logged to a timestamped file under `logs/`
- ✅ **Unit tests** (pytest) covering exceptions, logging, schema validation, and a couple of real bugs
  caught and fixed during the project's cleanup pass (see *Engineering Notes* below)
- 📦 **Reproducible environment** — dependencies and packaging live in one `pyproject.toml`, locked with
  [uv](https://docs.astral.sh/uv/) (`uv.lock`), no drift between a requirements file and setup.py

---

## 🗂️ Project Structure

```
moduler-project/
│
├── app.py                     # Streamlit app — single + batch prediction tabs
├── main.py                    # Runs the training pipeline end-to-end
├── pyproject.toml             # Packaging + dependencies (single source of truth, managed by uv)
├── uv.lock                    # Locked, reproducible dependency versions
├── schema.yml                 # Expected raw-data columns, checked at ingestion time
│
├── config/
│   └── config.yml             # Split ratio, random seed
│
├── data/
│   └── finalTrain.csv         # Source dataset — the ONLY thing tracked here, nothing generated
│
├── batch_prediction/
│   └── predict.py             # CLI: score a CSV of records with the trained model
│
├── scripts/
│   └── generate_project_structure.py   # Scaffolds a new project with this same layout
│
├── tests/                     # pytest unit tests
│
├── artifacts/                  # Everything generated at runtime — gitignored
│   ├── data_ingestion/<timestamp>/
│   │   ├── raw_data/raw.csv                 # snapshot of finalTrain.csv for this run
│   │   └── ingested_data/{train,test}.csv   # raw train/test split
│   ├── data_transformation/
│   │   ├── feature_engineered_data/{train,test}.csv  # readable, feeds app.py's dropdowns
│   │   ├── transformed_data/{train,test}.csv         # scaled numeric arrays fed to the model
│   │   └── processor/{processor,feature_engineering}.pkl
│   └── model_trainer/trained_model.pkl
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

This project follows [PEP 8](https://peps.python.org/pep-0008/) throughout:

| Element                       | Convention          | Example                                        |
|-------------------------------|----------------------|-------------------------------------------------|
| Packages / modules / folders  | `snake_case`         | `src/components`, `data_ingestion.py`           |
| Classes                        | `PascalCase`         | `DataIngestion`, `CustomException`              |
| Functions / variables          | `snake_case`         | `initiate_data_ingestion`, `train_data_path`    |
| Module-level constants          | `UPPER_SNAKE_CASE`   | `TARGET_COLUMN`, `RAW_DATA_PATH`                |

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

### 🌐 Launch the Streamlit app

Requires a trained model (run the command above first):

```bash
uv run streamlit run app.py
```

### 🧪 Run batch predictions from the command line

```bash
uv run python batch_prediction/predict.py \
  --input artifacts/data_transformation/feature_engineered_data/test.csv \
  --output batch_prediction/predictions.csv
```

(The same batch-scoring flow is also available inside the Streamlit app's **Batch Prediction** tab, as
a CSV upload with a downloadable results file.)

### ✅ Run tests

```bash
uv run pytest
```

Without uv: `pip install .` for the runtime dependencies, then `pip install pytest` for the test suite.

---

## 🔍 Sample Input Features

| Feature Name          | Description                                |
|------------------------|---------------------------------------------|
| `distance`             | Delivery distance in km, computed from restaurant/drop-off lat-long |
| `multiple_deliveries`  | Number of concurrent deliveries the rider was carrying |
| `Road_traffic_density` | Road/traffic status (`Low` → `Jam`, ordinal) |
| `Weather_conditions`   | Weather during delivery (`Sunny` → `Stormy`, ordinal) |
| `Time_Orderd_hour`     | Hour of day the order was placed           |
| `Type_of_order`        | Order type (Snack, Meal, Drinks, Buffet)   |

---

## 🧠 Engineering Notes

A few decisions worth calling out, since they came out of an actual cleanup/refactor pass rather than
being designed in from the start:

- **Every pipeline run is fully traceable.** Ingestion writes a timestamped snapshot of the raw split
  under `artifacts/data_ingestion/<timestamp>/` before anything downstream touches it, so a bad training
  run can always be traced back to exactly the data it saw.
- **`FeatureEngineering` is a real scikit-learn transformer** (`BaseEstimator` + `TransformerMixin`), not
  a stand-alone function — it drops into the same `Pipeline`/`ColumnTransformer` machinery as the rest of
  preprocessing, and explicitly declares itself `__sklearn_is_fitted__` since it's a stateless transform
  step with no parameters to learn.
- **Config over hardcoding.** Train/test split ratio and random seed are read from `config/config.yml`
  at runtime (falling back to sane defaults), rather than being magic numbers inside the ingestion code.
- **Schema validation on ingestion.** `schema.yml` declares the expected raw columns; ingestion checks
  the incoming data against it before doing anything else, so a malformed source file fails fast with a
  clear message instead of surfacing as a confusing error three stages later.
- A handful of real bugs were caught and fixed while restructuring this project: an exception-handling
  block in the original model trainer that wrapped a `def` statement instead of the method body (so
  training failures were silently never caught), a metrics call with its arguments swapped, and a
  constructor that accidentally turned every field into a 1-tuple via a stray trailing comma. All three
  now have regression tests in `tests/`.

---

## 📚 Tech Stack

- **Python 3.10+**
- **scikit-learn**, **XGBoost** — modeling and preprocessing pipelines
- **pandas**, **NumPy** — data handling
- **Streamlit** — interactive app (single + batch prediction)
- **pytest** — testing
- **uv** — dependency management and packaging

---

## 🗺️ Roadmap

Documented direction for where this project goes next (not yet implemented):

- **FastAPI backend** — split model-serving out from the Streamlit UI into a standalone API, so the
  model can be called independently of the app and loaded once at process startup instead of per script
  rerun.
- **Docker / Docker Compose** — containerize the API and the Streamlit frontend as separate services.
- **DVC** — version the dataset and trained model artifacts outside of git, with a declarative
  `dvc.yaml` pipeline instead of the current timestamp-folder convention.

---

## 📝 License

This project is licensed under the [Apache 2.0 License](LICENSE).
