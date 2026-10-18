SYSTEM ARCHITECTURE DIAGRAM

+---------------------------------------------------------------------------------------------------+
| MLOps End-to-End Pipeline |
+---------------------------------------------------------------------------------------------------+
| |
| [UCI ML Repository] |
| | |
| v |
| [Data Ingestion & EDA] --------> [Visualizations: Heatmap, Histograms, Class Balance] |
| | |
| v |
| [Feature Pipeline] ------------> Standard Scaling + Imputation Artifacts |
| | |
| v |
| [Model Training & CV] ---------> [MLflow Tracking Server] (Parameters, Metrics, ROC Curves) |
| | | |
| v v |
| [Model Serialization] ---------> [artifacts/models/*.pkl] |
| | |
| v |
| [GitHub Actions CI/CD] --------> Flake8 Linting ---> Pytest Suite ---> Automated Retraining |
| | |
| v |
| [Docker Engine (Minikube)] ----> Container Build (python:3.11-slim) |
| | |
| v |
| [Kubernetes Orchestration] |
| +---------------------------------------------------------------------------------------+ |
| | Minikube Cluster (Hyper-V Node) | |
| | | |
| | [LoadBalancer Service: Port 5000 -> NodePort 31110] | |
| | | | |
| | +-------------+-------------+ | |
| | | | | |
| | v v | |
| | [Pod 1: heart-disease-api] [Pod 2: heart-disease-api] | |
| +---------------------------------------------------------------------------------------+ |
| | |
| v |
| [Monitoring & Observability] --> Structured API Logs (`api_activity.log`) + Latency Tracking |
| |
+---------------------------------------------------------------------------------------------------+
