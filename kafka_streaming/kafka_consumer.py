import json
import time
import requests
import logging
from collections import deque
from kafka import KafkaConsumer
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataConsumer:
    """Consumes sensor data from Kafka, predicts RUL, and generates alerts."""
    
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='sensor-data',
                 api_url='http://localhost:8000/predict', window_size=30,
                 alert_threshold=25):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='latest',
            group_id='rul-predictor-group'
        )
        self.api_url = api_url
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Store sensor data for each engine
        self.engine_data = {}
        
        # Alert history
        self.alert_history = []
        
        logger.info(f"Consumer initialized for topic: {topic}")
        logger.info(f"API endpoint: {api_url}")
        logger.info(f"Window size: {window_size}")
        logger.info(f"Alert threshold: {alert_threshold} cycles")
    
    def process_sensor_data(self, data):
        """Process incoming sensor data and store in window."""
        engine_id = data['engine_id']
        
        if engine_id not in self.engine_data:
            self.engine_data[engine_id] = deque(maxlen=self.window_size)
        
        # Extract sensor values in the correct order
        sensor_values = []
        sensor_order = [
            'op_setting_1', 'op_setting_2', 'op_setting_3',
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
            'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
            'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
            'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
        ]
        
        for sensor in sensor_order:
            sensor_values.append(data['sensors'][sensor])
        
        # Add to window
        self.engine_data[engine_id].append(sensor_values)
        
        return len(self.engine_data[engine_id]) >= self.window_size
    
    def predict_rul(self, engine_id):
        """Call REST API to predict RUL for the given engine."""
        if engine_id not in self.engine_data:
            return None
        
        window_data = list(self.engine_data[engine_id])
        
        if len(window_data) < self.window_size:
            return None
        
        # Prepare data for API
        api_data = {
            "data": window_data,
            "feature_names": None  # Let API use default feature names
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=api_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['rul']
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call API: {e}")
            return None
    
    def check_alert(self, engine_id, rul, cycle):
        """Check if RUL is below threshold and generate alert."""
        if rul is None:
            return
        
        if rul <= self.alert_threshold:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'engine_id': engine_id,
                'cycle': cycle,
                'rul': rul,
                'severity': 'CRITICAL' if rul <= 10 else 'WARNING',
                'message': f"Engine {engine_id} has {rul:.1f} cycles remaining!"
            }
            
            self.alert_history.append(alert)
            
            # Log alert
            if alert['severity'] == 'CRITICAL':
                logger.critical(f"🚨 CRITICAL ALERT: {alert['message']}")
            else:
                logger.warning(f"⚠️  WARNING: {alert['message']}")
            
            # Print alert details
            print(f"\n{'='*60}")
            print(f"🚨 MAINTENANCE ALERT 🚨")
            print(f"{'='*60}")
            print(f"Engine ID: {alert['engine_id']}")
            print(f"Current Cycle: {alert['cycle']}")
            print(f"Remaining Useful Life: {alert['rul']:.1f} cycles")
            print(f"Severity: {alert['severity']}")
            print(f"Timestamp: {alert['timestamp']}")
            print(f"{'='*60}\n")
    
    def run(self):
        """Main consumer loop."""
        logger.info("Starting sensor data consumer...")
        logger.info("Waiting for messages...")
        
        try:
            for message in self.consumer:
                try:
                    data = message.value
                    engine_id = data['engine_id']
                    cycle = data['cycle']
                    
                    logger.info(f"Received data for engine {engine_id}, cycle {cycle}")
                    
                    # Process sensor data
                    window_ready = self.process_sensor_data(data)
                    
                    if window_ready:
                        # Predict RUL
                        rul = self.predict_rul(engine_id)
                        
                        if rul is not None:
                            logger.info(f"Engine {engine_id}, Cycle {cycle}: RUL = {rul:.1f}")
                            
                            # Check for alerts
                            self.check_alert(engine_id, rul, cycle)
                        else:
                            logger.warning(f"Failed to get RUL prediction for engine {engine_id}")
                    else:
                        logger.info(f"Building window for engine {engine_id}: {len(self.engine_data[engine_id])}/{self.window_size}")
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        finally:
            self.consumer.close()
            logger.info("Consumer closed")
    
    def get_statistics(self):
        """Get consumer statistics."""
        stats = {
            'total_alerts': len(self.alert_history),
            'critical_alerts': len([a for a in self.alert_history if a['severity'] == 'CRITICAL']),
            'warning_alerts': len([a for a in self.alert_history if a['severity'] == 'WARNING']),
            'engines_monitored': len(self.engine_data),
            'recent_alerts': self.alert_history[-10:] if self.alert_history else []
        }
        return stats


def main():
    """Main function to run the consumer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Sensor Data Consumer')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='sensor-data',
                       help='Kafka topic name')
    parser.add_argument('--api-url', default='http://localhost:8000/predict',
                       help='RUL prediction API URL')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Window size for RUL prediction')
    parser.add_argument('--alert-threshold', type=int, default=25,
                       help='RUL threshold for alerts')
    
    args = parser.parse_args()
    
    consumer = SensorDataConsumer(
        bootstrap_servers=args.bootstrap_servers.split(','),
        topic=args.topic,
        api_url=args.api_url,
        window_size=args.window_size,
        alert_threshold=args.alert_threshold
    )
    
    consumer.run()


if __name__ == "__main__":
    main() 