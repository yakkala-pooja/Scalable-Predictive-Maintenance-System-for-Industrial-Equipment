import json
import time
import random
import numpy as np
from kafka import KafkaProducer
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataProducer:
    """Simulates real-time sensor data streaming from industrial equipment."""
    
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='sensor-data'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.topic = topic
        self.engine_id = 1
        self.cycle = 1
        
        # Sensor configuration (matching CMAPSS dataset)
        self.sensor_config = {
            'op_setting_1': {'min': 0.0, 'max': 1.0, 'drift': 0.001},
            'op_setting_2': {'min': 0.0, 'max': 1.0, 'drift': 0.001},
            'op_setting_3': {'min': 0.0, 'max': 1.0, 'drift': 0.001},
            'sensor_1': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_2': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_3': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_4': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_5': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_6': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_7': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_8': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_9': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_10': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_11': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_12': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_13': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_14': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_15': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_16': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_17': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_18': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_19': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_20': {'min': 0.0, 'max': 1.0, 'drift': 0.002},
            'sensor_21': {'min': 0.0, 'max': 1.0, 'drift': 0.002}
        }
        
        # Initialize sensor values
        self.sensor_values = {}
        for sensor, config in self.sensor_config.items():
            self.sensor_values[sensor] = random.uniform(config['min'], config['max'])
    
    def generate_sensor_data(self):
        """Generate realistic sensor data with degradation patterns."""
        data = {
            'engine_id': self.engine_id,
            'cycle': self.cycle,
            'timestamp': datetime.now().isoformat(),
            'sensors': {}
        }
        
        # Simulate sensor degradation over time
        degradation_factor = min(1.0, self.cycle / 200)  # Gradual degradation
        
        for sensor, config in self.sensor_config.items():
            # Add drift and noise
            drift = config['drift'] * degradation_factor
            noise = random.gauss(0, 0.01)
            
            # Update sensor value
            self.sensor_values[sensor] += drift + noise
            
            # Keep within bounds
            self.sensor_values[sensor] = max(
                config['min'], 
                min(config['max'], self.sensor_values[sensor])
            )
            
            data['sensors'][sensor] = round(self.sensor_values[sensor], 6)
        
        return data
    
    def send_data(self, data):
        """Send sensor data to Kafka topic."""
        try:
            key = f"engine_{data['engine_id']}"
            future = self.producer.send(self.topic, key=key, value=data)
            future.get(timeout=10)  # Wait for send to complete
            logger.info(f"Sent data for engine {data['engine_id']}, cycle {data['cycle']}")
            return True
        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            return False
    
    def run(self, interval=1.0, max_cycles=300):
        """Run the producer for specified number of cycles."""
        logger.info(f"Starting sensor data producer for engine {self.engine_id}")
        logger.info(f"Will send data every {interval} seconds for {max_cycles} cycles")
        
        try:
            for cycle in range(1, max_cycles + 1):
                self.cycle = cycle
                data = self.generate_sensor_data()
                
                if self.send_data(data):
                    logger.info(f"Cycle {cycle}/{max_cycles} - Engine {self.engine_id}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Producer stopped by user")
        finally:
            self.producer.close()
            logger.info("Producer closed")


def main():
    """Main function to run the producer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Sensor Data Producer')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='sensor-data',
                       help='Kafka topic name')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Data sending interval in seconds')
    parser.add_argument('--max-cycles', type=int, default=300,
                       help='Maximum number of cycles to simulate')
    
    args = parser.parse_args()
    
    producer = SensorDataProducer(
        bootstrap_servers=args.bootstrap_servers.split(','),
        topic=args.topic
    )
    
    producer.run(interval=args.interval, max_cycles=args.max_cycles)


if __name__ == "__main__":
    main() 