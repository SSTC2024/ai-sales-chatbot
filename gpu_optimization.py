"""
Updated GPU Optimization for Natural Language AI Sales ChatBot
Compatible with latest sales_chatbot.py (NaturalLanguageSalesChatBot class)

Optimized for: RTX 4090 24GB + RTX 4070Ti Super 16GB
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer
import psutil
import GPUtil
import nvidia_ml_py3 as nvml
import threading
import queue
import time
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict, deque
import logging

class AdvancedGPUManager:
    """
    Advanced GPU management for NaturalLanguageSalesChatBot
    Optimized for heterogeneous GPU setups (RTX 4090 + RTX 4070Ti Super)
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.gpu_info = []
        self.device_allocation = {}
        self.performance_metrics = defaultdict(list)
        self.memory_tracker = {}
        
        # Initialize NVIDIA ML for detailed monitoring
        try:
            nvml.nvmlInit()
            self.nvml_available = True
        except:
            self.nvml_available = False
            
        self.discover_gpus()
        self.setup_optimal_allocation()
        self.start_monitoring()
        
    def discover_gpus(self):
        """Discover and analyze available GPUs"""
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - running in CPU mode")
            return
            
        gpu_count = torch.cuda.device_count()
        print(f"🔍 Discovered {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            
            gpu_data = {
                'id': i,
                'name': props.name,
                'memory_total': props.total_memory,
                'memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count,
                'device': f'cuda:{i}'
            }
            
            # Get detailed info if NVML available
            if self.nvml_available:
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Memory info
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_data['memory_free'] = mem_info.free
                    gpu_data['memory_used'] = mem_info.used
                    
                    # Temperature
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    gpu_data['temperature'] = temp
                    
                    # Power info
                    try:
                        power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        gpu_data['power_usage'] = power
                        
                        power_limit = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                        gpu_data['power_limit'] = power_limit
                    except:
                        gpu_data['power_usage'] = None
                        gpu_data['power_limit'] = None
                        
                except Exception as e:
                    print(f"Warning: Could not get detailed info for GPU {i}: {e}")
                    
            self.gpu_info.append(gpu_data)
            
        # Sort by memory size (largest first)
        self.gpu_info.sort(key=lambda x: x['memory_gb'], reverse=True)
        
        # Display GPU information
        print("\n🖥️  GPU Configuration:")
        for gpu in self.gpu_info:
            print(f"  GPU {gpu['id']}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_gb']:.1f}GB")
            print(f"    Compute: {gpu['compute_capability']}")
            if gpu.get('temperature'):
                print(f"    Temperature: {gpu['temperature']}°C")
            if gpu.get('power_usage'):
                print(f"    Power: {gpu['power_usage']:.1f}W / {gpu['power_limit']:.1f}W")
            print()
            
    def setup_optimal_allocation(self):
        """Setup optimal model allocation based on detected GPUs"""
        if not self.gpu_info:
            print("⚠️  No GPUs available - using CPU allocation")
            self.device_allocation = {
                'primary_device': 'cpu',
                'secondary_device': 'cpu',
                'llm_device': 'cpu',
                'embedding_device': 'cpu',
                'ocr_device': 'cpu'
            }
            return
            
        # Single GPU setup
        if len(self.gpu_info) == 1:
            primary_gpu = self.gpu_info[0]
            print(f"📱 Single GPU setup: {primary_gpu['name']}")
            
            self.device_allocation = {
                'primary_device': primary_gpu['device'],
                'secondary_device': primary_gpu['device'],
                'llm_device': primary_gpu['device'],
                'embedding_device': primary_gpu['device'],
                'ocr_device': primary_gpu['device']
            }
            
        # Multi-GPU setup (optimal for RTX 4090 + RTX 4070Ti Super)
        else:
            primary_gpu = self.gpu_info[0]  # Largest memory
            secondary_gpu = self.gpu_info[1]  # Second largest
            
            print(f"🚀 Multi-GPU setup detected:")
            print(f"  Primary: {primary_gpu['name']} ({primary_gpu['memory_gb']:.1f}GB)")
            print(f"  Secondary: {secondary_gpu['name']} ({secondary_gpu['memory_gb']:.1f}GB)")
            
            # Optimal allocation for RTX 4090 + RTX 4070Ti Super
            self.device_allocation = {
                'primary_device': primary_gpu['device'],      # RTX 4090 for LLM
                'secondary_device': secondary_gpu['device'],  # RTX 4070Ti Super for support
                'llm_device': primary_gpu['device'],         # Main language model
                'embedding_device': secondary_gpu['device'], # Sentence embeddings
                'ocr_device': secondary_gpu['device'],       # OCR processing
                'classification_device': secondary_gpu['device']  # Intent classification
            }
            
        print(f"\n⚙️  Device Allocation:")
        for role, device in self.device_allocation.items():
            print(f"  {role}: {device}")
            
    def get_optimal_model_config(self, model_type='llm'):
        """Get optimal configuration for different model types"""
        if not self.gpu_info:
            return {'device': 'cpu', 'quantization': False, 'max_memory': None}
            
        if model_type == 'llm':
            # Primary GPU configuration for LLM
            primary_gpu = self.gpu_info[0]
            memory_gb = primary_gpu['memory_gb']
            
            if memory_gb >= 20:  # RTX 4090 24GB
                return {
                    'device': self.device_allocation['llm_device'],
                    'quantization': False,  # Can run full precision
                    'model_size': 'large',  # Can handle Llama 2 13B
                    'max_memory': int(memory_gb * 0.8 * 1024**3),  # 80% of VRAM
                    'batch_size': 4,
                    'use_flash_attention': True
                }
            elif memory_gb >= 12:  # RTX 4070Ti Super 16GB
                return {
                    'device': self.device_allocation['llm_device'],
                    'quantization': True,   # Use 8-bit quantization
                    'model_size': 'medium', # Llama 2 7B with quantization
                    'max_memory': int(memory_gb * 0.7 * 1024**3),  # 70% of VRAM
                    'batch_size': 2,
                    'use_flash_attention': False
                }
            else:  # Smaller GPUs
                return {
                    'device': self.device_allocation['llm_device'],
                    'quantization': True,
                    'model_size': 'small',  # DialoGPT or similar
                    'max_memory': int(memory_gb * 0.6 * 1024**3),  # 60% of VRAM
                    'batch_size': 1,
                    'use_flash_attention': False
                }
                
        elif model_type == 'embedding':
            # Secondary GPU configuration for embeddings
            if len(self.gpu_info) > 1:
                secondary_gpu = self.gpu_info[1]
                return {
                    'device': self.device_allocation['embedding_device'],
                    'batch_size': 32,
                    'max_memory': int(secondary_gpu['memory_gb'] * 0.3 * 1024**3)  # 30% of VRAM
                }
            else:
                # Share primary GPU
                return {
                    'device': self.device_allocation['embedding_device'],
                    'batch_size': 16,
                    'max_memory': int(self.gpu_info[0]['memory_gb'] * 0.2 * 1024**3)  # 20% of VRAM
                }
                
        elif model_type == 'ocr':
            return {
                'device': self.device_allocation['ocr_device'],
                'batch_size': 8,
                'max_memory': int(1024**3)  # 1GB max for OCR
            }
            
    def create_quantization_config(self, model_config):
        """Create quantization configuration based on model config"""
        if not model_config.get('quantization', False):
            return None
            
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=False
        )
        
    def optimize_model_loading(self, model, model_config):
        """Optimize model loading based on configuration"""
        device = model_config['device']
        
        if device != 'cpu':
            # Enable optimizations for GPU
            if hasattr(model, 'half') and model_config.get('use_half_precision', True):
                model = model.half()
                
            # Enable gradient checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                
            # Optimize for inference
            model.eval()
            
        return model
        
    def start_monitoring(self):
        """Start background performance monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 GPU monitoring started")
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics every 5 seconds
                metrics = self.collect_gpu_metrics()
                timestamp = datetime.now()
                
                for gpu_id, metric in metrics.items():
                    self.performance_metrics[gpu_id].append({
                        'timestamp': timestamp,
                        'memory_used': metric['memory_used'],
                        'memory_total': metric['memory_total'],
                        'utilization': metric['utilization'],
                        'temperature': metric.get('temperature'),
                        'power_usage': metric.get('power_usage')
                    })
                    
                # Keep only last 100 measurements per GPU
                for gpu_id in self.performance_metrics:
                    if len(self.performance_metrics[gpu_id]) > 100:
                        self.performance_metrics[gpu_id] = self.performance_metrics[gpu_id][-100:]
                        
                time.sleep(5)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)
                
    def collect_gpu_metrics(self):
        """Collect current GPU metrics"""
        metrics = {}
        
        if not self.nvml_available:
            return metrics
            
        try:
            for gpu in self.gpu_info:
                gpu_id = gpu['id']
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                # Power
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except:
                    power = None
                    
                metrics[gpu_id] = {
                    'memory_used': mem_info.used,
                    'memory_total': mem_info.total,
                    'memory_percent': (mem_info.used / mem_info.total) * 100,
                    'utilization': util.gpu,
                    'memory_utilization': util.memory,
                    'temperature': temp,
                    'power_usage': power
                }
                
        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
            
        return metrics
        
    def get_performance_summary(self):
        """Get performance summary for all GPUs"""
        summary = {}
        current_metrics = self.collect_gpu_metrics()
        
        for gpu_id, gpu in enumerate(self.gpu_info):
            gpu_summary = {
                'name': gpu['name'],
                'memory_total_gb': gpu['memory_gb'],
                'device': gpu['device']
            }
            
            # Current metrics
            if gpu_id in current_metrics:
                metric = current_metrics[gpu_id]
                gpu_summary.update({
                    'memory_used_gb': metric['memory_used'] / (1024**3),
                    'memory_percent': metric['memory_percent'],
                    'utilization': metric['utilization'],
                    'temperature': metric['temperature'],
                    'power_usage': metric['power_usage']
                })
                
            # Historical averages
            if gpu_id in self.performance_metrics and self.performance_metrics[gpu_id]:
                recent_metrics = self.performance_metrics[gpu_id][-10:]  # Last 10 measurements
                
                avg_memory = np.mean([m['memory_used'] for m in recent_metrics]) / (1024**3)
                avg_util = np.mean([m['utilization'] for m in recent_metrics if m['utilization'] is not None])
                avg_temp = np.mean([m['temperature'] for m in recent_metrics if m['temperature'] is not None])
                
                gpu_summary.update({
                    'avg_memory_used_gb': avg_memory,
                    'avg_utilization': avg_util,
                    'avg_temperature': avg_temp
                })
                
            summary[gpu_id] = gpu_summary
            
        return summary
        
    def optimize_memory_usage(self):
        """Optimize memory usage across GPUs"""
        try:
            # Clear unused cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Force garbage collection
            import gc
            gc.collect()
            
            print("🧹 Memory optimization completed")
            
        except Exception as e:
            print(f"Error during memory optimization: {e}")
            
    def check_memory_availability(self, device, required_memory_gb):
        """Check if device has enough memory available"""
        if device == 'cpu':
            return True
            
        try:
            device_id = int(device.split(':')[1])
            current_metrics = self.collect_gpu_metrics()
            
            if device_id in current_metrics:
                available_memory = current_metrics[device_id]['memory_total'] - current_metrics[device_id]['memory_used']
                available_gb = available_memory / (1024**3)
                
                return available_gb >= required_memory_gb
                
        except Exception as e:
            print(f"Error checking memory availability: {e}")
            return False
            
        return False
        
    def get_recommended_models(self):
        """Get recommended models based on available hardware"""
        recommendations = {}
        
        if not self.gpu_info:
            recommendations['llm'] = ['microsoft/DialoGPT-small']
            recommendations['embedding'] = ['sentence-transformers/all-MiniLM-L6-v2']
            return recommendations
            
        primary_memory = self.gpu_info[0]['memory_gb']
        
        # LLM recommendations
        if primary_memory >= 20:  # RTX 4090 24GB
            recommendations['llm'] = [
                'meta-llama/Llama-2-13b-chat-hf',
                'meta-llama/Llama-2-7b-chat-hf',
                'mistralai/Mistral-7B-Instruct-v0.1'
            ]
        elif primary_memory >= 12:  # RTX 4070Ti Super 16GB
            recommendations['llm'] = [
                'meta-llama/Llama-2-7b-chat-hf',
                'mistralai/Mistral-7B-Instruct-v0.1',
                'microsoft/DialoGPT-large'
            ]
        else:  # Smaller GPUs
            recommendations['llm'] = [
                'microsoft/DialoGPT-medium',
                'microsoft/DialoGPT-large',
                'gpt2-medium'
            ]
            
        # Embedding model recommendations
        recommendations['embedding'] = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ]
        
        return recommendations
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        print("📊 GPU monitoring stopped")
        
    def export_performance_data(self, filename=None):
        """Export performance data to file"""
        if not filename:
            filename = f"gpu_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        export_data = {
            'gpu_info': self.gpu_info,
            'device_allocation': self.device_allocation,
            'performance_metrics': {}
        }
        
        # Convert performance metrics to JSON-serializable format
        for gpu_id, metrics in self.performance_metrics.items():
            export_data['performance_metrics'][str(gpu_id)] = []
            for metric in metrics:
                serializable_metric = {
                    'timestamp': metric['timestamp'].isoformat(),
                    'memory_used': metric['memory_used'],
                    'memory_total': metric['memory_total'],
                    'utilization': metric['utilization'],
                    'temperature': metric['temperature'],
                    'power_usage': metric['power_usage']
                }
                export_data['performance_metrics'][str(gpu_id)].append(serializable_metric)
                
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"📊 Performance data exported to {filename}")
        return filename

# Integration helper functions for NaturalLanguageSalesChatBot

def integrate_gpu_optimization(chatbot_instance):
    """
    Integrate GPU optimization with existing NaturalLanguageSalesChatBot instance
    """
    gpu_manager = AdvancedGPUManager()
    
    # Override device setup
    chatbot_instance.primary_device = gpu_manager.device_allocation['primary_device']
    chatbot_instance.secondary_device = gpu_manager.device_allocation['secondary_device']
    
    # Get optimal configurations
    llm_config = gpu_manager.get_optimal_model_config('llm')
    embedding_config = gpu_manager.get_optimal_model_config('embedding')
    
    # Store in chatbot instance
    chatbot_instance.gpu_manager = gpu_manager
    chatbot_instance.llm_config = llm_config
    chatbot_instance.embedding_config = embedding_config
    
    print("🔧 GPU optimization integrated with chatbot")
    return gpu_manager

def get_optimal_model_selection():
    """
    Get optimal model selection based on current hardware
    """
    gpu_manager = AdvancedGPUManager()
    recommendations = gpu_manager.get_recommended_models()
    
    print("🎯 Recommended models for your hardware:")
    print(f"LLM: {recommendations['llm'][0]}")
    print(f"Embedding: {recommendations['embedding'][0]}")
    
    return recommendations

if __name__ == "__main__":
    # Test GPU optimization
    print("🧪 Testing GPU Optimization...")
    
    gpu_manager = AdvancedGPUManager()
    
    print("\n📊 Performance Summary:")
    summary = gpu_manager.get_performance_summary()
    for gpu_id, info in summary.items():
        print(f"GPU {gpu_id}: {info['name']}")
        print(f"  Memory: {info.get('memory_used_gb', 0):.1f}GB / {info['memory_total_gb']:.1f}GB")
        print(f"  Utilization: {info.get('utilization', 0):.1f}%")
        if info.get('temperature'):
            print(f"  Temperature: {info['temperature']}°C")
    
    print("\n🎯 Model Recommendations:")
    recommendations = gpu_manager.get_recommended_models()
    for model_type, models in recommendations.items():
        print(f"{model_type.upper()}: {models[0]}")
    
    # Keep monitoring for 30 seconds then stop
    print("\n⏳ Monitoring for 30 seconds...")
    time.sleep(30)
    gpu_manager.stop_monitoring()
    
    # Export performance data
    gpu_manager.export_performance_data()
    print("✅ GPU optimization test completed")