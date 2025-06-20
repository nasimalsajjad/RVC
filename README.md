# MLOps Classification Project

This project implements a complete MLOps pipeline for a classification task, incorporating best practices for data versioning, model training, deployment, and monitoring.

## 🏗️ Project Structure

```
.
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data files
│   └── .gitignore             # Git ignore for data files
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data loading utilities
│   │   └── preprocessor.py    # Data preprocessing pipeline
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # Feature engineering pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Model training pipeline
│   │   ├── predict.py         # Model prediction utilities
│   │   └── evaluate.py        # Model evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Helper functions
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA notebooks
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── configs/
│   ├── data_config.yaml       # Data configuration
│   ├── model_config.yaml      # Model configuration
│   └── training_config.yaml   # Training configuration
├── .github/
│   └── workflows/
│       ├── train.yml          # Training workflow
│       └── deploy.yml         # Deployment workflow
├── docker/
│   ├── Dockerfile            # Main Dockerfile
│   └── docker-entrypoint.sh  # Container entrypoint script
├── docker-compose.yml        # Docker Compose configuration
├── .dockerignore            # Docker ignore file
├── dvc.yaml                 # DVC pipeline configuration
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── README.md                # Project documentation
```

## 🛠️ Technology Stack

### Core Components
- **ML Pipeline Orchestration**: DVC
- **CI/CD**: GitHub Actions
- **Artifact Storage**: Amazon S3
- **Model Deployment**: SageMaker Inference Endpoint
- **Monitoring**: CloudWatch (+ Grafana)
- **REST API**: SageMaker + API Gateway
- **Containerization**: Docker + Docker Compose

## 🐳 Containerization

### Development Environment
The project uses Docker for consistent development and deployment environments. Two main services are provided:

1. **ML Pipeline Service**
   - For running the ML pipeline
   - Includes all dependencies
   - Mounts local directories for development

2. **Jupyter Service**
   - For interactive development
   - Accessible at `http://localhost:8888`
   - Includes all ML dependencies

### Running with Docker

1. **Build the containers**
   ```bash
   docker-compose build
   ```

2. **Start the services**
   ```bash
   docker-compose up -d
   ```

3. **Run the ML pipeline**
   ```bash
   docker-compose exec ml-pipeline python src/models/train.py
   ```

4. **Access Jupyter Notebook**
   - Open `http://localhost:8888` in your browser
   - No password required in development

### Production Deployment
For production deployment, the Docker image is optimized for:
- Minimal size
- Security best practices
- Proper environment variable handling
- Efficient layer caching

## 🔄 MLOps Pipeline

### 1. Data Management
- Raw data stored in S3
- DVC for data versioning
- Automated data validation
- Data preprocessing pipeline

### 2. Model Development
- Modular code structure
- Experiment tracking
- Model versioning
- Automated testing

### 3. Training Pipeline
- Automated training workflow
- Model evaluation
- Performance metrics tracking
- Model artifact storage

### 4. Deployment
- Automated deployment to SageMaker
- Endpoint management
- A/B testing support
- Rollback capabilities

### 5. Monitoring
- Real-time performance monitoring
- Drift detection
- Error tracking
- Resource utilization

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- AWS Account
- GitHub Account
- DVC

### Installation

#### Using Docker (Recommended)
```bash
# Clone the repository
git clone <repository-url>

# Build and start containers
docker-compose up -d

# Install dependencies inside container
docker-compose exec ml-pipeline pip install -r requirements.txt

# Initialize DVC
docker-compose exec ml-pipeline dvc init
docker-compose exec ml-pipeline dvc remote add -d s3remote s3://your-bucket
```

#### Manual Installation
```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init
dvc remote add -d s3remote s3://your-bucket
```

### Configuration
1. Set up AWS credentials
2. Configure DVC remote storage
3. Update configuration files in `configs/`
4. Create `.env` file with required environment variables

## 🔧 Development Workflow

1. **Data Processing**
   ```bash
   docker-compose exec ml-pipeline dvc run -n prepare_data -d src/data/preprocessor.py -o data/processed python src/data/preprocessor.py
   ```

2. **Feature Engineering**
   ```bash
   docker-compose exec ml-pipeline dvc run -n create_features -d src/features/feature_engineering.py -o data/features python src/features/feature_engineering.py
   ```

3. **Model Training**
   ```bash
   docker-compose exec ml-pipeline dvc run -n train_model -d src/models/train.py -o models/model.pkl python src/models/train.py
   ```

4. **Deployment**
   ```bash
   docker-compose exec ml-pipeline dvc run -n deploy -d src/deploy/deploy.py python src/deploy/deploy.py
   ```

## 📊 Monitoring

### CloudWatch Metrics
- Request count
- Latency
- Error rates
- Model performance metrics

### Grafana Dashboards
- Real-time monitoring
- Historical trends
- Alerting

## 🔐 Security

- AWS IAM roles and policies
- Secure credential management
- Network security
- Data encryption
- Docker security best practices

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Contact

For any questions or concerns, please open an issue in the repository. #   R V C  
 