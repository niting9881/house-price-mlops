# MLOps Grafana Monitoring Setup

This directory contains a complete Grafana-based monitoring solution for the MLOps house price prediction application. The setup includes Prometheus for metrics collection, Grafana for visualization, and custom metrics exporters for ML-specific monitoring.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML API        │    │  Model Exporter │    │   Prometheus    │
│  (Port 8000)    │───▶│  (Port 8001)    │───▶│  (Port 9090)    │
│                 │    │                 │    │                 │
│ • Predictions   │    │ • Model Metrics │    │ • Metrics Store │
│ • API Metrics   │    │ • Data Quality  │    │ • Time Series   │
│ • Health Check  │    │ • Drift Detection│   │ • Alerting      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │    Grafana      │
                                               │  (Port 3000)    │
                                               │                 │
                                               │ • Dashboards    │
                                               │ • Alerts        │
                                               │ • Visualizations│
                                               └─────────────────┘
```

## 📊 Monitoring Features

### 1. **Model Performance Monitoring**
- Real-time accuracy, precision, recall, F1-score tracking
- Performance degradation detection
- Model inference latency monitoring
- Prediction volume analytics

### 2. **Data Drift Detection**
- Statistical drift detection per feature
- P-value tracking for drift significance
- Automated drift alerts
- Feature-level drift visualization

### 3. **Data Quality Monitoring**
- Data completeness tracking
- Missing value detection
- Outlier identification
- Schema validation monitoring

### 4. **API Performance Monitoring**
- Request/response latency tracking
- API throughput monitoring
- Error rate tracking
- Active request monitoring

### 5. **Business Metrics**
- Daily prediction counts
- Prediction value distributions
- Service uptime tracking
- Resource utilization monitoring

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.11+ with required packages
- Running ML API service

### 1. Start Monitoring Stack

**On Windows:**
```bash
cd monitoring/grafana
./setup-monitoring.bat
```

**On Linux/Mac:**
```bash
cd monitoring/grafana
chmod +x setup-monitoring.sh
./setup-monitoring.sh
```

**Manual Setup:**
```bash
cd monitoring/grafana
docker-compose up -d --build
```

### 2. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | None |
| Model Metrics | http://localhost:8001/metrics | None |

### 3. View Dashboards

1. Open Grafana at http://localhost:3000
2. Login with admin/admin123
3. Navigate to Dashboards → Browse
4. Select "MLOps Model Monitoring Dashboard" or "Data Drift Monitoring Dashboard"

## 📈 Available Dashboards

### 1. MLOps Model Monitoring Dashboard
- **Model Performance**: Real-time accuracy, precision, recall metrics
- **Data Drift Alert**: Number of features with detected drift
- **API Performance**: Request rates and prediction throughput
- **Data Quality**: Overall quality scores and completeness
- **Response Times**: API latency and prediction latency percentiles
- **Feature Drift**: P-values for individual features

### 2. Data Drift Monitoring Dashboard
- **Feature Drift Over Time**: P-value trends for all features
- **Current Drift Status**: Table showing drift status per feature
- **Drift Summary**: Total features with drift detected
- **Drift Distribution**: Pie chart of drift vs. no-drift features
- **Drift Trends**: Historical drift detection patterns

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_PORT` | 8001 | Port for model metrics exporter |
| `METRICS_DB_PATH` | /app/data/metrics.db | Path to metrics database |
| `EXPORT_INTERVAL` | 30 | Metrics export interval (seconds) |

### Customizing Dashboards

1. **Edit Existing Dashboards:**
   - Modify JSON files in `dashboards/` directory
   - Restart Grafana: `docker-compose restart grafana`

2. **Add New Dashboards:**
   - Create new JSON files in `dashboards/` directory
   - Follow Grafana dashboard JSON format
   - Dashboards are auto-loaded on startup

3. **Modify Metrics:**
   - Edit `model_exporter.py` to add new metrics
   - Rebuild container: `docker-compose up -d --build model-exporter`

### Adding Custom Metrics

1. **Define Prometheus Metrics:**
```python
from prometheus_client import Counter, Gauge, Histogram

custom_metric = Gauge('custom_metric_name', 'Description', ['label1', 'label2'])
```

2. **Update Metrics in Exporter:**
```python
def update_custom_metrics(self):
    # Your metric collection logic
    custom_metric.labels(label1='value1', label2='value2').set(metric_value)
```

3. **Add to Export Cycle:**
```python
def run_export_cycle(self):
    # ... existing metrics ...
    self.update_custom_metrics()
```

## 🚨 Alerting Setup

### Grafana Alerts

1. **Navigate to Alerting → Alert Rules**
2. **Create New Rule:**
   - Query: Select metric and threshold
   - Condition: Define alert condition
   - Action: Configure notification channels

### Example Alert Rules

**Model Performance Degradation:**
```
WHEN max() OF query(A, 5m, now) IS BELOW 0.85
```

**Data Drift Detection:**
```
WHEN max() OF query(drift_columns_count, 1m, now) IS ABOVE 0
```

**API Latency Alert:**
```
WHEN avg() OF query(api_request_duration_seconds, 5m, now) IS ABOVE 2.0
```

## 🔍 Metrics Reference

### Model Performance Metrics
- `model_accuracy{model_name, version}` - Model accuracy score
- `model_precision{model_name, version}` - Model precision score  
- `model_recall{model_name, version}` - Model recall score
- `model_f1_score{model_name, version}` - Model F1 score

### Data Drift Metrics
- `data_drift_detected{feature}` - Binary drift detection flag
- `data_drift_pvalue{feature}` - Statistical p-value for drift test
- `drift_columns_count` - Total number of features with drift

### Data Quality Metrics
- `data_quality_score` - Overall data quality score (0-1)
- `missing_values_count{column}` - Count of missing values per column
- `data_completeness{column}` - Data completeness percentage per column
- `outliers_count{column}` - Count of outliers per column

### API Metrics
- `api_requests_total{method, endpoint, status}` - Total API requests
- `api_request_duration_seconds{method, endpoint}` - API request latency histogram
- `prediction_latency_seconds` - Model prediction latency histogram
- `daily_predictions_total` - Total predictions made
- `api_active_requests` - Current number of active API requests

### Business Metrics
- `prediction_value_distribution` - Distribution of prediction values
- `model_inference_time_seconds` - Model inference time histogram
- `model_memory_usage_bytes` - Model memory usage

## 🛠️ Troubleshooting

### Common Issues

1. **Grafana Not Loading:**
   - Check if port 3000 is available
   - Verify Docker containers are running: `docker-compose ps`
   - Check logs: `docker-compose logs grafana`

2. **No Metrics Data:**
   - Ensure ML API is running and generating metrics
   - Check model exporter logs: `docker-compose logs model-exporter`
   - Verify Prometheus can scrape metrics: http://localhost:9090/targets

3. **Dashboard Not Showing:**
   - Check dashboard provisioning: `docker-compose logs grafana`
   - Verify dashboard JSON syntax
   - Restart Grafana: `docker-compose restart grafana`

4. **Metrics Exporter Failing:**
   - Check if metrics database exists
   - Verify Python dependencies in container
   - Check exporter logs: `docker-compose logs model-exporter`

### Debug Commands

```bash
# Check all services status
docker-compose ps

# View logs for specific service
docker-compose logs -f grafana
docker-compose logs -f prometheus
docker-compose logs -f model-exporter

# Test metrics endpoint
curl http://localhost:8001/metrics

# Test Prometheus targets
curl http://localhost:9090/api/v1/targets

# Access container shell
docker-compose exec grafana /bin/bash
docker-compose exec prometheus /bin/sh
```

### Performance Tuning

1. **Prometheus Storage:**
   - Adjust retention time in `prometheus.yml`
   - Monitor disk usage: `docker system df`

2. **Grafana Performance:**
   - Reduce dashboard refresh rates
   - Limit time ranges for heavy queries
   - Use appropriate aggregation intervals

3. **Metrics Export:**
   - Adjust `EXPORT_INTERVAL` based on needs
   - Optimize database queries in exporter
   - Consider metric sampling for high-volume data

## 📚 Additional Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)
- [Prometheus Metrics Types](https://prometheus.io/docs/concepts/metric_types/)

## 🤝 Contributing

To add new monitoring features:

1. Fork the repository
2. Create feature branch
3. Add metrics to `model_exporter.py`
4. Create/update dashboard JSON
5. Update documentation
6. Submit pull request

## 📝 License

This monitoring setup is part of the MLOps house price prediction project and follows the same license terms.
