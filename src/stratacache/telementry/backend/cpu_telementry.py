from dataclasses import dataclass

from stratacache.telementry.telementry import StrataTierStats, StrataTierTelementry, StrataTierType
from stratacache.telementry.utils import human_readable_size

import psutil
import logging

logger = logging.getLogger(__name__)

class SystemCPUDetector:
    class CPUStats:
        cpu_usage: float = 0.0
        cpu_freq: float = 0.0
        
    @staticmethod
    def get_cpu_stats() -> CPUStats:
        """
        Get system CPU statistics using psutil.
        This method is cross-platform and doesn't require subprocess calls.

        Returns:
            CPUStats object containing CPU usage percentage and frequency in MHz.
        """
        try:
            stats = SystemCPUDetector.CPUStats()
            stats.cpu_usage = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            if cpu_freq is not None:
                stats.cpu_freq = cpu_freq.current
            return stats

        except Exception as e:
            logger.warning(f"Failed to get system CPU stats using psutil: {e}")
            return SystemCPUDetector.CPUStats()

class SystemMemoryDetector:
    class MemoryStats:
        free_mem: int = 0
        used_mem: int = 0
        total_mem: int = 0
    
    @staticmethod
    def get_mem_stats() -> MemoryStats:
        """
        Get system memory statistics using psutil.
        This method is cross-platform and doesn't require subprocess calls.

        Returns:
            MemoryStats object containing free, used, and total memory in bytes.
        """
        try:
            memory = psutil.virtual_memory()
            stats = SystemMemoryDetector.MemoryStats()
            stats.free_mem = memory.available
            stats.used_mem = memory.used
            stats.total_mem = memory.total
            return stats

        except Exception as e:
            logger.warning(f"Failed to get system available memory using psutil: {e}")
            return SystemMemoryDetector.MemoryStats()

@dataclass
class StrataCPUStats(StrataTierStats):
    cpu_usage: float = 0.0
    cpu_freq: float = 0.0

    free_mem_gb: float = 0.0
    used_mem_gb: float = 0.0
    total_mem_gb: float = 0.0
    
class StrataCPUTelemetry(StrataTierTelementry):
    def __init__(self):
        super().__init__(StrataTierType.CPU)
        self._stats = StrataCPUStats()
    
    def process_op(self, op_type: str, **kwargs):
        logger.info(f"Processing CPU operation: op_type={op_type}, kwargs={kwargs}")
        
        cpu_stats = SystemCPUDetector.get_cpu_stats()
        self._stats.cpu_usage = cpu_stats.cpu_usage
        self._stats.cpu_freq = cpu_stats.cpu_freq
        
        mem_stats = SystemMemoryDetector.get_mem_stats()
        self._stats.free_mem_gb = mem_stats.free_mem / (1024**3)
        self._stats.used_mem_gb = mem_stats.used_mem / (1024**3)
        self._stats.total_mem_gb = mem_stats.total_mem / (1024**3)        
    
    def print_stats(self):
        logger.info(f"CPU Stats: cpu_usage={self._stats.cpu_usage:.2f}%, cpu_freq={self._stats.cpu_freq:.2f}MHz, total_ops={self._stats.total_ops}, total_read_ops={self._stats.total_read_ops}, total_write_ops={self._stats.total_write_ops}, "
                    f"byte_read={human_readable_size(self._stats.byte_read)}, byte_written={human_readable_size(self._stats.byte_written)}, "
                    f"avg_read_throughput={human_readable_size(self._stats.avg_read_throughput)}/s, max_read_throughput={human_readable_size(self._stats.max_read_throughput)}/s, min_read_throughput={human_readable_size(self._stats.min_read_throughput)}/s, "
                    f"avg_read_latency_us={self._stats.avg_read_latency_us:.2f}us, max_read_latency_us={self._stats.max_read_latency_us:.2f}us, min_read_latency_us={self._stats.min_read_latency_us:.2f}us, "
                    f"avg_write_throughput={human_readable_size(self._stats.avg_write_throughput)}/s, max_write_throughput={human_readable_size(self._stats.max_write_throughput)}/s, min_write_throughput={human_readable_size(self._stats.min_write_throughput)}/s, "
                    f"avg_write_latency_us={self._stats.avg_write_latency_us:.2f}us, max_write_latency_us={self._stats.max_write_latency_us:.2f}us, min_write_latency_us={self._stats.min_write_latency_us:.2f}us, "
                    f"avg_throughput={human_readable_size(self._stats.avg_throughput)}/s, max_throughput={human_readable_size(self._stats.max_throughput)}/s, min_throughput={human_readable_size(self._stats.min_throughput)}/s, "
                    f"avg_latency_us={self._stats.avg_latency_us:.2f}us, max_latency_us={self._stats.max_latency_us:.2f}us, min_latency_us={self._stats.min_latency_us:.2f}us")
    
    def dump_stats(self) -> dict:
        """Dump CPU-specific stats for export."""
        base_stats = super().dump_stats()
        # Add CPU-specific metrics
        base_stats["cpu_usage"] = self._stats.cpu_usage
        base_stats["cpu_freq"] = self._stats.cpu_freq
        base_stats["free_mem_gb"] = self._stats.free_mem_gb
        base_stats["used_mem_gb"] = self._stats.used_mem_gb
        base_stats["total_mem_gb"] = self._stats.total_mem_gb
        return base_stats