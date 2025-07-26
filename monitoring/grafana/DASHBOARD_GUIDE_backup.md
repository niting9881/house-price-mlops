# MLOps Dashboard & Alert Configuration Guide

## 🎯 Dashboard Access
**Grafana URL:** http://localhost:3000
**Username:** `admin`
**Password:** `admin123`

## 📊 Pre-configured Dashboards

### 1. MLOps Model Monitoring Dashboard
- **Location:** Home → Dashboards → "MLOps Model Monitoring Dashboard"
- **Description:** Main monitoring dashboard with comprehensive ML metrics

#### Dashboard Panels:
1. **Model Performance Metrics**
   - Model accuracy over time
   - RMSE (Root Mean Square Error)
   - R² Score
   - Current values updated every 30 seconds

2. **Data Drift Alert**
   - Statistical drift detection
   - P-value monitoring for feature shifts
   - Drift status indicators

3. **API Performance**
   - Request rate and latency
   - Response time distribution
   - Error rate monitoring

4. **Data Quality Metrics**
   - Data validation scores
   - Quality trend analysis

5. **Response Time Metrics**
   - API response time histograms
   - Performance percentiles

6. **Feature Drift P-values**
   - Individual feature drift monitoring
   - Statistical significance tracking

### 2. Data Drift Dashboard
- **Location:** Home → Dashboards → "Data Drift Detection"
- **Focus:** Specialized drift monitoring and analysis

## 🚨 Alert Configuration

### Current Alert Rules (Active):
1. **Model Accuracy Below Threshold**
   - **Trigger:** Model accuracy < 0.85 (85%)
   - **Severity:** Critical
   - **Duration:** Fires after 1 minute
   - **Description:** Alerts when model performance degrades

2. **Data Drift Detected**
   - **Trigger:** Drift p-value < 0.05
   - **Severity:** Warning
   - **Duration:** Fires after 2 minutes
   - **Description:** Statistical drift in feature distributions

3. **API Response Time High**
   - **Trigger:** Average response time > 1.0 seconds
   - **Severity:** Warning
   - **Duration:** Fires after 3 minutes
   - **Description:** API performance degradation

4. **System CPU Usage High**
   - **Trigger:** CPU usage > 80%
   - **Severity:** Warning
   - **Duration:** Fires after 5 minutes
   - **Description:** System resource constraints

5. **Model Predictions Stalled**
   - **Trigger:** No predictions for 10 minutes
   - **Severity:** Critical
   - **Duration:** Fires after 10 minutes
   - **Description:** Service availability issues

## 🔧 Dashboard Navigation Steps

### Step 1: Login to Grafana
1. Open http://localhost:3000
2. Enter credentials: admin / admin123
3. Click "Log in"

### Step 2: Access Dashboards
1. Click "Dashboards" in left sidebar (folder icon)
2. Select "MLOps Model Monitoring Dashboard"
3. Observe real-time metrics updating

### Step 3: Explore Dashboard Features
- **Time Range:** Use time picker (top right) to adjust viewing period
- **Refresh Rate:** Set auto-refresh interval (default: 30s)
- **Panel Zoom:** Click on any panel to zoom in
- **Data Drill-down:** Hover over metrics for detailed values

### Step 4: View Alert Rules
1. Go to "Alerting" → "Alert Rules" in left sidebar
2. Review configured alert conditions
3. Check current alert status

### Step 5: Configure Notifications (Optional)
1. Go to "Alerting" → "Contact Points"
2. Add email/webhook endpoints
3. Configure notification policies

## 📈 Current Metrics Status
```
Model Accuracy: 87.0% ✅ (Above 85% threshold)
Bedrooms Drift P-value: 0.092 ✅ (Above 0.05 threshold)
Bathrooms Drift P-value: 0.067 ✅ (Above 0.05 threshold)
CPU Usage: 47.2% ✅ (Below 80% threshold)
```

## 🔍 Monitoring Endpoints

### Prometheus Metrics
- **URL:** http://localhost:9090
- **Purpose:** Raw metrics collection and querying

### Model Metrics Exporter
- **URL:** http://localhost:8001/metrics
- **Purpose:** ML-specific metrics generation

### FastAPI ML Service
- **URL:** http://localhost:8000
- **Purpose:** Model prediction service

## 🎛️ Customization Options

### Adding New Panels
1. Click "Add panel" in dashboard
2. Configure data source (Prometheus)
3. Write PromQL queries for new metrics
4. Set visualization type and thresholds

### Modifying Alert Thresholds
1. Go to "Alerting" → "Alert Rules"
2. Click "Edit" on any rule
3. Modify threshold values
4. Update evaluation intervals

### Dashboard Export/Import
- **Export:** Dashboard settings → Export
- **Import:** Dashboards → Import → Upload JSON

## 🎯 Key Dashboard Features to Explore

1. **Real-time Updates:** All metrics refresh automatically
2. **Interactive Charts:** Click and drag to zoom
3. **Alert Annotations:** See alert firing times on charts
4. **Multi-panel Views:** Compare related metrics side-by-side
5. **Historical Analysis:** Review performance trends over time

## 🚀 Next Steps

1. **Customize Dashboards:** Add panels for business-specific metrics
2. **Set Up Notifications:** Configure email/Slack alerts
3. **Create Custom Alerts:** Add rules for domain-specific thresholds
4. **Monitor Trends:** Use historical data for capacity planning
5. **Automate Responses:** Integrate with incident management systems

---
*This monitoring setup provides comprehensive observability for your MLOps pipeline with real-time metrics, alerting, and historical analysis capabilities.*
