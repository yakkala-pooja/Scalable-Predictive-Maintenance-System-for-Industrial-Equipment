import torch
import os
import logging
from typing import Dict, Any, Optional, Tuple
import subprocess
import platform

logger = logging.getLogger(__name__)


class HardwareManager:
    """Manages hardware detection and configuration for optimal performance."""
    
    def __init__(self):
        """Initialize hardware manager."""
        self.device_info = self._detect_hardware()
        self.logger = logging.getLogger('hardware')
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware."""
        info = {
            'cpu': self._detect_cpu(),
            'gpu': self._detect_gpu(),
            'tpu': self._detect_tpu(),
            'memory': self._detect_memory(),
            'platform': platform.system(),
            'architecture': platform.machine()
        }
        
        logger.info(f"Hardware detection completed: {info}")
        return info
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information."""
        try:
            import psutil
            cpu_info = {
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'usage_percent': psutil.cpu_percent(interval=1),
                'available': True
            }
        except ImportError:
            # Fallback without psutil
            cpu_info = {
                'count': os.cpu_count() or 1,
                'count_logical': os.cpu_count() or 1,
                'frequency': None,
                'usage_percent': None,
                'available': True
            }
        
        logger.info(f"CPU detected: {cpu_info['count']} cores")
        return cpu_info
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information."""
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'memory': []
        }
        
        try:
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['count'] = torch.cuda.device_count()
                
                for i in range(gpu_info['count']):
                    device_props = torch.cuda.get_device_properties(i)
                    device_info = {
                        'id': i,
                        'name': device_props.name,
                        'memory_total': device_props.total_memory,
                        'memory_free': torch.cuda.get_device_properties(i).total_memory,
                        'compute_capability': f"{device_props.major}.{device_props.minor}",
                        'multi_processor_count': device_props.multi_processor_count
                    }
                    gpu_info['devices'].append(device_info)
                    
                    # Get current memory usage
                    try:
                        torch.cuda.set_device(i)
                        memory_allocated = torch.cuda.memory_allocated(i)
                        memory_reserved = torch.cuda.memory_reserved(i)
                        device_info['memory_allocated'] = memory_allocated
                        device_info['memory_reserved'] = memory_reserved
                        device_info['memory_free'] = device_info['memory_total'] - memory_reserved
                    except Exception as e:
                        logger.warning(f"Could not get memory info for GPU {i}: {e}")
                
                logger.info(f"GPU detected: {gpu_info['count']} devices")
            else:
                logger.info("No CUDA GPU available")
                
        except Exception as e:
            logger.warning(f"Error detecting GPU: {e}")
        
        return gpu_info
    
    def _detect_tpu(self) -> Dict[str, Any]:
        """Detect TPU information."""
        tpu_info = {
            'available': False,
            'count': 0,
            'devices': []
        }
        
        try:
            # Check for TPU environment variables
            tpu_env_vars = ['TPU_NAME', 'TPU_IP_ADDRESS', 'TPU_WORKER_ID']
            tpu_env_present = any(os.getenv(var) for var in tpu_env_vars)
            
            if tpu_env_present:
                tpu_info['available'] = True
                tpu_info['count'] = 1  # Usually 1 TPU per process
                
                # Get TPU device info
                device_info = {
                    'name': os.getenv('TPU_NAME', 'unknown'),
                    'ip_address': os.getenv('TPU_IP_ADDRESS', 'unknown'),
                    'worker_id': os.getenv('TPU_WORKER_ID', 'unknown')
                }
                tpu_info['devices'].append(device_info)
                
                logger.info("TPU detected")
            else:
                logger.info("No TPU available")
                
        except Exception as e:
            logger.warning(f"Error detecting TPU: {e}")
        
        return tpu_info
    
    def _detect_memory(self) -> Dict[str, Any]:
        """Detect system memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'available_gb': memory.available / (1024**3)
            }
        except ImportError:
            memory_info = {
                'total': None,
                'available': None,
                'used': None,
                'percent': None,
                'available_gb': None
            }
        
        return memory_info
    
    def get_optimal_device_config(self, model_size_mb: float = 100) -> Dict[str, Any]:
        """Get optimal device configuration based on available hardware and model size."""
        config = {
            'device_type': 'cpu',
            'device_count': 1,
            'strategy': None,
            'mixed_precision': False,
            'num_workers': 0,
            'pin_memory': False
        }
        
        # Check if TPU is available
        if self.device_info['tpu']['available']:
            config.update({
                'device_type': 'tpu',
                'device_count': self.device_info['tpu']['count'],
                'strategy': 'tpu_spawn',
                'mixed_precision': True
            })
            logger.info("Using TPU for training")
            
        # Check if GPU is available
        elif self.device_info['gpu']['available']:
            gpu_count = self.device_info['gpu']['count']
            
            # Estimate GPU memory requirements (rough estimate)
            gpu_memory_gb = min([device['memory_free'] / (1024**3) for device in self.device_info['gpu']['devices']])
            model_memory_gb = model_size_mb / 1024  # Convert MB to GB
            
            if gpu_memory_gb > model_memory_gb * 2:  # Need 2x model size for training
                if gpu_count > 1:
                    config.update({
                        'device_type': 'gpu',
                        'device_count': gpu_count,
                        'strategy': 'ddp',
                        'mixed_precision': True,
                        'num_workers': min(4, self.device_info['cpu']['count']),
                        'pin_memory': True
                    })
                    logger.info(f"Using {gpu_count} GPUs with DDP strategy")
                else:
                    config.update({
                        'device_type': 'gpu',
                        'device_count': 1,
                        'strategy': None,
                        'mixed_precision': True,
                        'num_workers': min(4, self.device_info['cpu']['count']),
                        'pin_memory': True
                    })
                    logger.info("Using single GPU")
            else:
                logger.warning(f"GPU memory ({gpu_memory_gb:.1f}GB) may be insufficient for model ({model_memory_gb:.1f}GB)")
                config.update({
                    'device_type': 'cpu',
                    'device_count': 1,
                    'strategy': None,
                    'num_workers': min(8, self.device_info['cpu']['count']),
                    'pin_memory': False
                })
                logger.info("Falling back to CPU due to insufficient GPU memory")
                
        # Use CPU
        else:
            config.update({
                'device_type': 'cpu',
                'device_count': 1,
                'strategy': None,
                'num_workers': min(8, self.device_info['cpu']['count']),
                'pin_memory': False
            })
            logger.info("Using CPU for training")
        
        return config
    
    def get_pytorch_lightning_config(self, model_size_mb: float = 100) -> Dict[str, Any]:
        """Get PyTorch Lightning configuration for optimal performance."""
        device_config = self.get_optimal_device_config(model_size_mb)
        
        pl_config = {
            'accelerator': device_config['device_type'],
            'devices': device_config['device_count'],
            'num_workers': device_config['num_workers'],
            'pin_memory': device_config['pin_memory'],
            'precision': '16-mixed' if device_config['mixed_precision'] else '32',
        }
        
        # Add strategy for multi-device training
        if device_config['strategy']:
            pl_config['strategy'] = device_config['strategy']
        
        # Special handling for TPU
        if device_config['device_type'] == 'tpu':
            pl_config['accelerator'] = 'tpu'
            pl_config['devices'] = 8  # TPU typically has 8 cores
        
        logger.info(f"PyTorch Lightning config: {pl_config}")
        return pl_config
    
    def get_optimal_batch_size(self, model_size_mb: float = 100, base_batch_size: int = 32) -> int:
        """Get optimal batch size based on available hardware."""
        device_config = self.get_optimal_device_config(model_size_mb)
        
        if device_config['device_type'] == 'tpu':
            # TPU works best with batch sizes that are multiples of 128
            return max(128, base_batch_size * 4)
        
        elif device_config['device_type'] == 'gpu':
            gpu_count = device_config['device_count']
            # Scale batch size with number of GPUs
            return base_batch_size * gpu_count
        
        else:  # CPU
            cpu_count = self.device_info['cpu']['count']
            # For CPU, use smaller batch size but more workers
            return max(16, base_batch_size // 2)
    
    def get_memory_optimization_settings(self) -> Dict[str, Any]:
        """Get memory optimization settings."""
        settings = {
            'gradient_accumulation_steps': 1,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'automatic_optimization': True
        }
        
        # Check available memory
        if self.device_info['memory']['available_gb']:
            available_gb = self.device_info['memory']['available_gb']
            
            if available_gb < 8:  # Less than 8GB RAM
                settings.update({
                    'gradient_accumulation_steps': 4,
                    'accumulate_grad_batches': 4
                })
                logger.info("Low memory detected, enabling gradient accumulation")
        
        return settings
    
    def log_hardware_summary(self):
        """Log a summary of available hardware."""
        logger.info("=" * 60)
        logger.info("HARDWARE SUMMARY")
        logger.info("=" * 60)
        
        # CPU
        cpu = self.device_info['cpu']
        logger.info(f"CPU: {cpu['count']} cores available")
        
        # GPU
        gpu = self.device_info['gpu']
        if gpu['available']:
            logger.info(f"GPU: {gpu['count']} devices available")
            for i, device in enumerate(gpu['devices']):
                memory_gb = device['memory_total'] / (1024**3)
                logger.info(f"  GPU {i}: {device['name']} ({memory_gb:.1f}GB)")
        else:
            logger.info("GPU: Not available")
        
        # TPU
        tpu = self.device_info['tpu']
        if tpu['available']:
            logger.info(f"TPU: {tpu['count']} devices available")
            for device in tpu['devices']:
                logger.info(f"  TPU: {device['name']}")
        else:
            logger.info("TPU: Not available")
        
        # Memory
        memory = self.device_info['memory']
        if memory['available_gb']:
            logger.info(f"System Memory: {memory['available_gb']:.1f}GB available")
        
        logger.info("=" * 60)


def get_hardware_manager() -> HardwareManager:
    """Get a singleton hardware manager instance."""
    if not hasattr(get_hardware_manager, '_instance'):
        get_hardware_manager._instance = HardwareManager()
    return get_hardware_manager._instance


def optimize_for_hardware(model_size_mb: float = 100, base_batch_size: int = 32) -> Dict[str, Any]:
    """Get optimized configuration for the current hardware."""
    hw_manager = get_hardware_manager()
    
    # Log hardware summary
    hw_manager.log_hardware_summary()
    
    # Get optimal configurations
    device_config = hw_manager.get_optimal_device_config(model_size_mb)
    pl_config = hw_manager.get_pytorch_lightning_config(model_size_mb)
    batch_size = hw_manager.get_optimal_batch_size(model_size_mb, base_batch_size)
    memory_settings = hw_manager.get_memory_optimization_settings()
    
    # Combine all configurations
    config = {
        'device_config': device_config,
        'pl_config': pl_config,
        'batch_size': batch_size,
        'memory_settings': memory_settings
    }
    
    logger.info(f"Optimized configuration: {config}")
    return config


if __name__ == "__main__":
    # Test hardware detection
    hw_manager = HardwareManager()
    hw_manager.log_hardware_summary()
    
    # Test optimization
    config = optimize_for_hardware(model_size_mb=200, base_batch_size=32)
    print(f"Optimal configuration: {config}") 