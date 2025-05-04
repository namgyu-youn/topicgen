# topicgen

[![Python3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
<a href="https://github.com/namgyu-youn/topicgen/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/namgyu-youn/topicgen?color=2b9348"></a>
![pre-commit](https://github.com/namgyu-youn/topicgen/actions/workflows/pre-commit.yaml/badge.svg)


This tool aims to predict relevant **GitHub topics** for repositories by analyzing their content. It collects repository data via the `GitHub API`, processes descriptive text and `README` files, and utilizes a `BERT-based multi-label classifier` to suggest appropriate topics. The system includes complete data collection and model training pipelines, with support for exporting trained PyTorch models for deployment.

## Project Structure
### Project Overview
![project_architecture](https://github.com/user-attachments/assets/dbe27e6e-0a94-4151-8733-4f6e3c16b800)

### Data Collection Pipeline - [Sample Database](https://github.com/Namgyu-Youn/topicgen/blob/main/data/topicgen.db)
![data_collection_pipeline](https://github.com/user-attachments/assets/868250d0-7309-483b-9b35-cdcae71a96a1)

### Model Training Pipeline
![training_pipeline](https://github.com/user-attachments/assets/d75ab603-fa3f-4d46-abc0-5c270571af23)

## ✨ Features

### GPU Acceleration
This project now supports GPU acceleration for faster model training and inference:
- Uses PyTorch with CUDA 12.1 support
- Automatically detects and uses available GPUs
- Falls back to CPU when no GPU is available
- Docker images with GPU support via NVIDIA Container Toolkit

### Core Features
- **Collects GitHub repository data** (metadata, topics, READMEs) via GitHub API
- **Analyzes repository content** to predict relevant topics using ML models
- Trains a BERT-based **multi-label classifier** for topic prediction
- Stores repository and topic data in SQLite for efficient retrieval
- Exports trained PyTorch models for production deployment

## 🚩 How to use?

```bash
git clone https://github.com/Namgyu-Youn/topicgen.git
cd topicgen
```

### Option 1: Using uv (Recommended)
```bash
pip install uv
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# For CPU-only installation
uv pip install -r requirements.txt

# For GPU support (CUDA 12.1)
uv pip install -r requirements.txt

# Data Collection Pipeline
python -m topicgen.pipelines.data_collection_pipeline --min-stars 1000 --language python --max-repos 500

# Model Training Pipeline
python -m topicgen.pipelines.model_training_pipeline --base-model bert-base-uncased --num-epochs 5
```

### Option 2: Using Docker
```bash
# Build the Docker image (CPU version)
docker build -t github-topic-generator .

# Build with GPU support
docker build -t github-topic-generator:gpu .

# Run data collection pipeline
docker run github-topic-generator python -m topicgen.pipelines.data_collection_pipeline

# Run model training pipeline with GPU support
docker run --gpus all github-topic-generator:gpu python -m topicgen.pipelines.model_training_pipeline

# Or with CPU only
docker run github-topic-generator python -m topicgen.pipelines.model_training_pipeline
```

### Option 3: Standard Python Setup
```bash
python -m venv env

# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate

pip install -r requirements.txt

# Data Collection Pipeline
python -m topicgen.pipelines.data_collection_pipeline

# Model Training Pipeline
python -m topicgen.pipelines.model_training_pipeline
```

## 🧐 Introduction about gradio UI

1. Enter GitHub URL
2. Select the main, sub category that best matches your repository
3. Click "Generate Topics" to get your results
4. Enjoy generated topics('#')! It can be used like this.


## 👥 Contribution guide : [CONTRIBUTING.md](https://github.com/Namgyu-Youn/github-topic-generator/blob/main/CONTRIBUTING.md)
Thanks for your interest. I always enjoy meaningful collaboration. <br/>
Do you have any question or bug?? Then submit **ISSUE**. You can also use awesome labels(🏷️).