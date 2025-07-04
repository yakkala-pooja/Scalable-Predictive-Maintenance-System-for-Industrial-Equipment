# Predictive Maintenance System - Deployment Guide

This guide explains how to deploy and use the complete Scalable Predictive Maintenance System for Industrial Equipment.

## 🚀 Quick Start

### Prerequisites

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **Trained model** (checkpoints directory)
- **Prepared data** (transformer_data directory)

### 1. Prepare Your Environment

First, ensure you have trained a model and prepared your data:

```bash
# Train the model (if not already done)
python models/run_pipeline.py --max_epochs 10

# Or skip training if you already have checkpoints
python models/run_pipeline.py --skip_data_prep --skip_training
```

### 2. Deploy the System

```bash
# Make the deployment script executable
chmod +x deploy.sh

# Deploy the complete system
./deploy.sh
```

This will start:
- **Kafka & Zookeeper** - Message streaming infrastructure
- **Predictive Maintenance API** - REST API for RUL predictions
- **Kafka Producer** - Simulates real-time sensor data
- **Kafka Consumer** - Processes data and generates alerts
- **Kafka UI** - Web interface for monitoring

### 3. Access the System

Once deployed, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Kafka UI**: http://localhost:8080
- **API Health Check**: http://localhost:8000/

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sensor Data   │───▶│   Kafka Topic   │───▶│  Kafka Consumer │
│   Simulator     │    │  (sensor-data)  │    │  (RUL Predictor)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Kafka UI      │    │  Alert Generator│
                       │  (Monitoring)   │    │  (RUL < 25)     │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  REST API       │
                       │  (/predict)     │
                       └─────────────────┘
```

## 🔧 API Usage

### RUL Prediction Endpoint

**POST** `/predict`

Predicts the Remaining Useful Life (RUL) for given sensor data.

**Request Body:**
```json
{
  "data": [
    [0.1, 0.2, 0.3, ...],  // 24 sensor values for time step 1
    [0.2, 0.3, 0.4, ...],  // 24 sensor values for time step 2
    ...
  ],  // Array of 30 time steps (window_size)
  "feature_names": null  // Optional: specify feature names
}
```

**Response:**
```json
{
  "rul": 45.2,
  "details": null
}
```

### Example API Call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
      [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    ]
  }'
```

## 📈 Monitoring and Alerts

### Real-Time Alerts

The system automatically generates alerts when:
- **Warning**: RUL ≤ 25 cycles
- **Critical**: RUL ≤ 10 cycles

Alerts appear in the console with detailed information:
```
============================================================
🚨 MAINTENANCE ALERT 🚨
============================================================
Engine ID: 1
Current Cycle: 245
Remaining Useful Life: 8.3 cycles
Severity: CRITICAL
Timestamp: 2024-01-15T10:30:45.123456
============================================================
```

### Kafka UI Monitoring

Access http://localhost:8080 to:
- Monitor message flow
- View topic statistics
- Inspect message content
- Monitor consumer lag

## 🛠️ Management Commands

### Deployment Script Commands

```bash
# Deploy the system
./deploy.sh

# Stop all services
./deploy.sh stop

# Restart all services
./deploy.sh restart

# View logs
./deploy.sh logs

# Check service status
./deploy.sh status

# Clean up (removes volumes)
./deploy.sh clean
```

### Docker Compose Commands

```bash
# Start specific service
docker-compose up -d predictive-maintenance-api

# View logs for specific service
docker-compose logs -f kafka-consumer

# Scale services
docker-compose up -d --scale kafka-producer=3

# Update and restart
docker-compose pull && docker-compose up -d
```

## 🔍 Troubleshooting

### Common Issues

1. **API not responding**
   ```bash
   # Check if API container is running
   docker-compose ps predictive-maintenance-api
   
   # Check API logs
   docker-compose logs predictive-maintenance-api
   ```

2. **Kafka connection issues**
   ```bash
   # Check Kafka status
   docker-compose ps kafka
   
   # Check Kafka logs
   docker-compose logs kafka
   ```

3. **Model loading errors**
   ```bash
   # Ensure checkpoints exist
   ls -la checkpoints/
   
   # Check model file permissions
   docker-compose exec predictive-maintenance-api ls -la /app/checkpoints/
   ```

### Performance Tuning

1. **Increase Kafka throughput**
   ```yaml
   # In docker-compose.yml
   kafka:
     environment:
       KAFKA_NUM_NETWORK_THREADS: 8
       KAFKA_NUM_IO_THREADS: 8
   ```

2. **Scale API instances**
   ```bash
   docker-compose up -d --scale predictive-maintenance-api=3
   ```

3. **Adjust consumer parallelism**
   ```bash
   docker-compose up -d --scale kafka-consumer=2
   ```

## 📊 Evaluation and Testing

### Run Precision Evaluation

```bash
# Evaluate model precision for critical failures
python evaluate_precision.py --model-path checkpoints/tft-epoch=10-val_loss=1000.0000.ckpt

# This will generate:
# - precision_evaluation_report.txt
# - precision_evaluation_plots.png
```

### Test Streaming Pipeline

```bash
# Test producer only
python kafka_streaming/kafka_producer.py --max-cycles 50

# Test consumer only
python kafka_streaming/kafka_consumer.py --alert-threshold 30
```

## 🔒 Security Considerations

1. **Network Security**
   - Expose only necessary ports
   - Use internal Docker networks
   - Implement API authentication if needed

2. **Data Security**
   - Encrypt sensitive data in transit
   - Implement proper access controls
   - Regular security updates

3. **Production Deployment**
   - Use HTTPS for API endpoints
   - Implement rate limiting
   - Add monitoring and alerting
   - Use secrets management

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NASA CMAPSS Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs: `docker-compose logs -f`
3. Check service status: `docker-compose ps`
4. Verify prerequisites are met

---

**Note**: This system is designed for demonstration and development purposes. For production deployment, additional security, monitoring, and scaling considerations should be implemented. 