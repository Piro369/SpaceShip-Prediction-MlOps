# SpaceShip-Prediction-MLOps

An end-to-end Machine Learning Operations (MLOps) project designed to predict whether Spaceship passengers are transported to another dimension. 

This project goes beyond basic model training by implementing a fully automated, production-ready pipeline utilizing modern MLOps tools and cloud infrastructure.

## 🚀 Architecture Blueprint

This pipeline is structured to handle data versioning, experiment tracking, continuous integration, and cloud deployment:
* **Data & Pipeline Versioning:** DVC (Data Version Control) with AWS S3 remote storage.
* **Experiment Tracking:** MLflow for logging metrics, parameters, and model artifacts.
* **Web Application:** FastAPI for serving the serialized inference pipeline.
* **CI/CD:** GitHub Actions (Self-hosted runner) for automated testing, containerization, and deployment.
* **Cloud Deployment:** Docker containerization, AWS ECR (Elastic Container Registry) for image storage, and AWS EC2 for live serving.

## 📂 Project Organization

```text
├── .dvc                   <- DVC configuration and remote storage settings
├── app                    <- FastAPI application code and API routing
├── data                   <- Tracked via DVC (Not pushed to Git)
│   ├── external           <- Data from third party sources
│   ├── interim            <- Intermediate data that has been transformed
│   ├── processed          <- The final, canonical data sets for modeling
│   └── raw                <- The original, immutable data dump
├── docs                   <- A default mkdocs project
├── models                 <- Serialized models and sklearn pipelines (.joblib/.pkl)
├── notebooks              <- Jupyter notebooks for EDA and initial experimentation
├── references             <- Data dictionaries, manuals, and all other explanatory materials
├── reports                <- Generated analysis (HTML, PDF, LaTeX)
│   └── figures            <- Generated graphics and figures to be used in reporting
├── src                    <- Source code for data processing and model training
│   ├── __init__.py        
│   ├── split_data.py      <- Script to split raw data into train/test sets
│   ├── preprocessor.py    <- Script to build the preprocessing and inference pipeline
│   └── model_training.py  <- Script to train the model and log via MLflow
├── .dvcignore             <- Files and directories to be ignored by DVC
├── .env                   <- Environment variables (AWS keys, etc. - DO NOT COMMIT)
├── .gitignore             <- Files and directories to be ignored by Git
├── dvc.lock               <- DVC lock file defining the exact pipeline state
├── dvc.yaml               <- DVC pipeline stages (split, preprocess, train)
├── Makefile               <- Makefile with convenience commands
└── README.md              <- The top-level README for developers
```

## 🛠️ Getting Started

### 1. Clone the repository and setup environment
```bash
git clone <your-repo-url>
cd SpaceShip-Prediction-MLOps
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts ctivate`
pip install -r requirements.txt
```

### 2. Configure AWS Credentials
Ensure your AWS credentials are set up for DVC to pull from S3:
```bash
aws configure
```

### 3. Pull the Data
Pull the versioned datasets and models from the remote S3 bucket:
```bash
dvc pull
```

## 🔄 Reproducing the ML Pipeline

Whenever you change data or code in the `src` folder, you can reproduce the entire pipeline (splitting, preprocessing, and training) with a single command:
```bash
dvc repro
```
*Note: DVC is smart and will only re-run the stages that have changed!*

## 📊 Experiment Tracking

To view the tracked metrics (accuracy, loss), parameters, and registered models, spin up the MLflow UI:
```bash
mlflow ui
```
Navigate to `http://localhost:5000` in your browser.

## 🌐 Running the FastAPI Application Locally

Once your model pipeline is saved (e.g., `models/model_pipeline.joblib`), you can serve it via FastAPI:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Navigate to `http://localhost:8000/docs` to test the API endpoints.

## 🐳 Docker Containerization

To build and run the complete application inside a Docker container (simulating the final EC2 deployment):
```bash
docker build -t spaceship-mlops-app .
docker run -p 8000:8000 spaceship-mlops-app
```
