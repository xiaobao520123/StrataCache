from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

import copy

import logging
import threading
import queue
from datetime import datetime

# process id
import os

from stratacache.telementry.utils import thread_safe, human_readable_size
from stratacache.telementry.exporters import (
    ExporterManager, 
    FileExporter, 
    WanDBExporter, 
    TelemetrySnapshot
)

import stratacache.config as config

logger = logging.getLogger(__name__)

# Optional WanDB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class StrataTierType(Enum):
    GPU = 0
    CPU = 1
    CXL = 2
    NIXL = 3
    DISK = 4
    
    UNKNOWN = 99

@dataclass
class StrataTierStats():
    total_ops: int = 0
    total_read_ops: int = 0
    total_write_ops: int = 0
    
    byte_read: int = 0
    byte_written: int = 0
    
    size: int = 0
    
    avg_read_throughput: float = 0.0
    max_read_throughput: float = 0.0
    min_read_throughput: float = 0.0
    avg_read_latency_us: float = 0.0
    max_read_latency_us: float = 0.0
    min_read_latency_us: float = 0.0
    
    avg_write_throughput: float = 0.0
    max_write_throughput: float = 0.0
    min_write_throughput: float = 0.0
    avg_write_latency_us: float = 0.0
    max_write_latency_us: float = 0.0
    min_write_latency_us: float = 0.0
    
    avg_throughput: float = 0.0
    max_throughput: float = 0.0
    min_throughput: float = 0.0
    avg_latency_us: float = 0.0
    max_latency_us: float = 0.0
    min_latency_us: float = 0.0
    

class StrataTierTelementry(ABC):
    def __init__(self, tier: StrataTierType):
        self._lock = threading.RLock()
        self._tier = tier
        StrataTelemetry.GetOrCreate().register_tier(tier, self)
        
    def get_stats(self) -> StrataTierStats:
        with self._lock:
            return copy.deepcopy(self._stats)
    
    @abstractmethod
    def process_op(self, op_type: str, **kwargs):
        raise NotImplementedError("process_op must be implemented by subclasses")
    
    @abstractmethod
    def print_stats(self):
        raise NotImplementedError("print_stats must be implemented by subclasses")
    
    def dump_stats(self) -> dict:
        """Dump tier stats as dictionary for serialization.
        Can be overridden by subclasses for custom dump logic.
        """
        tier_stats = self.get_stats()
        return tier_stats.__dict__


@dataclass
class StrataSystemStats():
    total_ops: int = 0
    total_read_ops: int = 0
    total_write_ops: int = 0
    
    byte_read: int = 0
    byte_written: int = 0
    
    size: int = 0
    
    avg_read_throughput: float = 0.0
    max_read_throughput: float = 0.0
    min_read_throughput: float = 0.0
    avg_read_latency_us: float = 0.0
    max_read_latency_us: float = 0.0
    min_read_latency_us: float = 0.0
    
    avg_write_throughput: float = 0.0
    max_write_throughput: float = 0.0
    min_write_throughput: float = 0.0
    avg_write_latency_us: float = 0.0
    max_write_latency_us: float = 0.0
    min_write_latency_us: float = 0.0
    
    avg_throughput: float = 0.0
    max_throughput: float = 0.0
    min_throughput: float = 0.0
    avg_latency_us: float = 0.0
    max_latency_us: float = 0.0
    min_latency_us: float = 0.0
    
    def print_stats(self):
        logger.info(f"System Stats: total_ops={self.total_ops}, total_read_ops={self.total_read_ops}, total_write_ops={self.total_write_ops}, "
                    f"byte_read={human_readable_size(self.byte_read)}, byte_written={human_readable_size(self.byte_written)}, "
                    f"avg_read_throughput={human_readable_size(self.avg_read_throughput)}/s, max_read_throughput={human_readable_size(self.max_read_throughput)}/s, min_read_throughput={human_readable_size(self.min_read_throughput)}/s, "
                    f"avg_read_latency_us={self.avg_read_latency_us:.2f}ms, max_read_latency_us={self.max_read_latency_us:.2f}ms, min_read_latency_us={self.min_read_latency_us:.2f}ms, "
                    f"avg_write_throughput={human_readable_size(self.avg_write_throughput)}/s, max_write_throughput={human_readable_size(self.max_write_throughput)}/s, min_write_throughput={human_readable_size(self.min_write_throughput)}/s, "
                    f"avg_write_latency_us={self.avg_write_latency_us:.2f}ms, max_write_latency_us={self.max_write_latency_us:.2f}ms, min_write_latency_us={self.min_write_latency_us:.2f}ms, "
                    f"avg_throughput={human_readable_size(self.avg_throughput)}/s, max_throughput={human_readable_size(self.max_throughput)}/s, min_throughput={human_readable_size(self.min_throughput)}/s, "
                    f"avg_latency_us={self.avg_latency_us:.2f}ms, max_latency_us={self.max_latency_us:.2f}ms, min_latency_us={self.min_latency_us:.2f}ms")
    

class StrataTelemetry:
    """_summary_
    StrataTelemetry collects telemetry data for StrataCache.
    It provides telementry from two perspective: system-overall and by-tier.
    """
    
    def __init__(self):
        self._system_stats_lock = threading.RLock()
        self._system_stats = StrataSystemStats()
        self._per_tier_telementry: dict[StrataTierType, StrataTierTelementry] = {}
        self._message_queue = queue.Queue()
        
        # Initialize exporter manager with file exporter by default
        self._exporter_manager = ExporterManager()
        # self._exporter_manager.add_exporter(FileExporter())
        
        if config.get_wandb_enabled() == True:
            self.enable_wandb(
                entity=config.get_wandb_entity(),
                project=config.get_wandb_project(),
                name=f"{config.get_wandb_run_name()}-{os.getpid()}",
            )
            self._exporter_manager.add_exporter(WanDBExporter())
        
        threading.Thread(target=self._process_messages, daemon=True).start()
        threading.Thread(target=self._dump_stats, daemon=True).start()
    
    _instance = None

    @staticmethod
    @thread_safe
    def GetOrCreate() -> "StrataTelemetry":
        if StrataTelemetry._instance is None:
            StrataTelemetry._instance = StrataTelemetry()
        return StrataTelemetry._instance
    
    def on_tier_op_async(self, tier: StrataTierType, op_type: str, **kwargs):
        self._message_queue.put_nowait((tier, op_type, kwargs))

    def get_system_stats(self) -> StrataSystemStats:
        with self._system_stats_lock:
            return copy.deepcopy(self._system_stats)
    
    @thread_safe
    def get_tier_stats(self, tier: StrataTierType) -> StrataTierStats:
        telementry = self._per_tier_telementry.get(tier, None)
        if telementry is None:
            raise ValueError(f"Invalid tier type: {tier}")
        return telementry.get_stats()
    
    @thread_safe
    def register_tier(self, tier: StrataTierType, telementry: StrataTierTelementry):
        logging.info(f"Registering telemetry for tier: {tier}")
        self._per_tier_telementry[tier] = telementry
    
    def enable_wandb(self, entity: str, project: str = "stratacache", name: str = None, config: dict = None):
        if not WANDB_AVAILABLE:
            logger.error("WanDB not installed. Install it with: pip install wandb")
            raise ImportError("WanDB not installed. Install it with: pip install wandb")
        
        if entity is None or entity.strip() == "":
            logger.error("WanDB entity is required to enable WanDB integration")
            raise ValueError("WanDB entity is required to enable WanDB integration")
        
        try:
            if wandb.run is not None:
                logger.info("WanDB already initialized")
                return True
            
            init_config = {"module": "stratacache.telementry"}
            if config:
                init_config.update(config)
            
            wandb.init(
                entity=entity,
                project=project,
                name=name or "stratacache-experiment",
                config=init_config
            )
            logger.info(f"✓ WanDB initialized: project={project}, name={wandb.run.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WanDB: {e}")
            return False
    
    def disable_wandb(self):
        """Finalize WanDB run.
        
        This closes the WanDB run and should be called before application exit.
        Also removes WanDBExporter from the exporter manager.
        """
        # Remove WanDB exporter
        self._exporter_manager.remove_exporter(WanDBExporter)
        
        # Close WanDB run
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                wandb.finish()
                logger.info("WanDB run finalized")
            except Exception as e:
                logger.warning(f"Error closing WanDB: {e}")
    
    def add_exporter(self, exporter):
        """Add a telemetry exporter.
        
        Args:
            exporter: TelemetryExporter instance (FileExporter, WanDBExporter, etc.)
            
        Example:
            from stratacache.telementry.exporters import PrometheusExporter
            telemetry = StrataTelemetry.GetOrCreate()
            telemetry.add_exporter(PrometheusExporter())
        """
        self._exporter_manager.add_exporter(exporter)
    
    def remove_exporter(self, exporter_class):
        """Remove all exporters of a given class.
        
        Args:
            exporter_class: Class to remove (e.g., WanDBExporter, FileExporter)
            
        Example:
            from stratacache.telementry.exporters import FileExporter
            telemetry = StrataTelemetry.GetOrCreate()
            telemetry.remove_exporter(FileExporter)
        """
        self._exporter_manager.remove_exporter(exporter_class)
    
    def _process_messages(self):
        logger.info("Starting telemetry message processing thread.")
        
        while True:
            tier, op_type, kwargs = self._message_queue.get()
            
            logger.info(f"Processing telemetry message: tier={tier}, op_type={op_type}, kwargs={kwargs}")
            
            # Generic tier metrics
            latency_us = kwargs.get("latency_us", 0.0)
            size = kwargs.get("size", 0)
            throughput = (size / latency_us * 1_000_000 if latency_us > 0 else 0)
            
            # Update system stats
            ## NOTICE: System stats should have higher level of stats report. Here we duplicate tier stats.
            with self._system_stats_lock:
                sys_stats = self._system_stats
                sys_stats.total_ops += 1
                sys_stats.avg_latency_us = ((sys_stats.avg_latency_us * (sys_stats.total_ops - 1)) + latency_us) / sys_stats.total_ops
                sys_stats.max_latency_us = max(sys_stats.max_latency_us, latency_us)
                sys_stats.min_latency_us = latency_us if sys_stats.min_latency_us == 0.0 else min(sys_stats.min_latency_us, latency_us)
                sys_stats.avg_throughput = ((sys_stats.avg_throughput * (sys_stats.total_ops - 1)) + throughput) / sys_stats.total_ops
                sys_stats.max_throughput = max(sys_stats.max_throughput, throughput)
                sys_stats.min_throughput = throughput if sys_stats.min_throughput == 0.0 else min(sys_stats.min_throughput, throughput)
                
                if op_type == "load":
                    sys_stats.total_read_ops += 1
                    sys_stats.byte_read += size
                    sys_stats.avg_read_latency_us = ((sys_stats.avg_read_latency_us * (sys_stats.total_read_ops - 1)) + latency_us) / sys_stats.total_read_ops
                    sys_stats.max_read_latency_us = max(sys_stats.max_read_latency_us, latency_us)
                    sys_stats.min_read_latency_us = latency_us if sys_stats.min_read_latency_us == 0.0 else min(sys_stats.min_read_latency_us, latency_us)
                    sys_stats.avg_read_throughput = ((sys_stats.avg_read_throughput * (sys_stats.total_read_ops - 1)) + throughput) / sys_stats.total_read_ops
                    sys_stats.max_read_throughput = max(sys_stats.max_read_throughput, throughput)
                    sys_stats.min_read_throughput = throughput if sys_stats.min_read_throughput == 0.0 else min(sys_stats.min_read_throughput, throughput)
                elif op_type == "store":
                    sys_stats.total_write_ops += 1
                    sys_stats.byte_written += size
                    sys_stats.avg_write_latency_us = ((sys_stats.avg_write_latency_us * (sys_stats.total_write_ops - 1)) + latency_us) / sys_stats.total_write_ops
                    sys_stats.max_write_latency_us = max(sys_stats.max_write_latency_us, latency_us)
                    sys_stats.min_write_latency_us = latency_us if sys_stats.min_write_latency_us == 0.0 else min(sys_stats.min_write_latency_us, latency_us)
                    sys_stats.avg_write_throughput = ((sys_stats.avg_write_throughput * (sys_stats.total_write_ops - 1)) + throughput) / sys_stats.total_write_ops
                    sys_stats.max_write_throughput = max(sys_stats.max_write_throughput, throughput)
                    sys_stats.min_write_throughput = throughput if sys_stats.min_write_throughput == 0.0 else min(sys_stats.min_write_throughput, throughput)
                    sys_stats.size += size
                elif op_type == "delete":
                    logger.info(f"Delete operation: size={size}, latency_us={latency_us}")
                    sys_stats.size -= size
                else:
                    logger.warning(f"Unknown operation type: {op_type}")
                sys_stats.print_stats()
                
            if tier not in self._per_tier_telementry:
                logger.warning(f"No telemetry handler for tier: {tier}")
                continue

            telementry = self._per_tier_telementry[tier]
            # Update tier stats
            with telementry._lock:
                tier_stats = telementry._stats
                tier_stats.total_ops += 1
                
                tier_stats.avg_latency_us = ((tier_stats.avg_latency_us * (tier_stats.total_ops - 1)) + latency_us) / tier_stats.total_ops
                tier_stats.max_latency_us = max(tier_stats.max_latency_us, latency_us)
                tier_stats.min_latency_us = latency_us if tier_stats.min_latency_us == 0.0 else min(tier_stats.min_latency_us, latency_us)
                
                tier_stats.avg_throughput = ((tier_stats.avg_throughput * (tier_stats.total_ops - 1)) + throughput) / tier_stats.total_ops
                tier_stats.max_throughput = max(tier_stats.max_throughput, throughput)
                tier_stats.min_throughput = throughput if tier_stats.min_throughput == 0.0 else min(tier_stats.min_throughput, throughput)
                
                if op_type == "load":
                    tier_stats.total_read_ops += 1
                    tier_stats.byte_read += size
                    
                    tier_stats.avg_read_latency_us = ((tier_stats.avg_read_latency_us * (tier_stats.total_read_ops - 1)) + latency_us) / tier_stats.total_read_ops
                    tier_stats.max_read_latency_us = max(tier_stats.max_read_latency_us, latency_us)
                    tier_stats.min_read_latency_us = latency_us if tier_stats.min_read_latency_us == 0.0 else min(tier_stats.min_read_latency_us, latency_us)
                    
                    tier_stats.avg_read_throughput = ((tier_stats.avg_read_throughput * (tier_stats.total_read_ops - 1)) + throughput) / tier_stats.total_read_ops
                    tier_stats.max_read_throughput = max(tier_stats.max_read_throughput, throughput)
                    tier_stats.min_read_throughput = throughput if tier_stats.min_read_throughput == 0.0 else min(tier_stats.min_read_throughput, throughput)
                elif op_type == "store":
                    tier_stats.total_write_ops += 1
                    tier_stats.byte_written += size
                    
                    tier_stats.avg_write_latency_us = ((tier_stats.avg_write_latency_us * (tier_stats.total_write_ops - 1)) + latency_us) / tier_stats.total_write_ops
                    tier_stats.max_write_latency_us = max(tier_stats.max_write_latency_us, latency_us)
                    tier_stats.min_write_latency_us = latency_us if tier_stats.min_write_latency_us == 0.0 else min(tier_stats.min_write_latency_us, latency_us)
                    
                    tier_stats.avg_write_throughput = ((tier_stats.avg_write_throughput * (tier_stats.total_write_ops - 1)) + throughput) / tier_stats.total_write_ops
                    tier_stats.max_write_throughput = max(tier_stats.max_write_throughput, throughput)
                    tier_stats.min_write_throughput = throughput if tier_stats.min_write_throughput == 0.0 else min(tier_stats.min_write_throughput, throughput)
                    
                    tier_stats.size += size
                elif op_type == "delete":
                    logger.info(f"Delete operation: size={size}, latency_us={latency_us}")
                    tier_stats.size -= size
                else:
                    logger.warning(f"Unknown operation type: {op_type}")
                
                # Call tier-specific processing
                telementry.process_op(op_type, **kwargs)
                telementry.print_stats()
    
    def _dump_stats(self):
        """Dump telemetry stats to all registered exporters.
        """
        dump_interval = 1  # seconds
        
        while True:
            try:
                threading.Event().wait(dump_interval)
                
                timestamp = datetime.now().isoformat()
                
                # Build system stats dict
                system_stats_dict = self.get_system_stats().__dict__
                
                # Build per-tier stats dict
                tier_stats_dict = {}
                for tier, telementry in self._per_tier_telementry.items():
                    tier_stats_dict[tier.name] = telementry.dump_stats()
                
                # Create telemetry snapshot
                snapshot = TelemetrySnapshot(
                    timestamp=timestamp,
                    system_stats=system_stats_dict,
                    tier_stats=tier_stats_dict
                )
                
                # Export via all registered exporters
                results = self._exporter_manager.export(snapshot)
                
                # Log export results
                success_count = sum(1 for v in results.values() if v)
                if success_count > 0:
                    logger.debug(f"Exported stats: {success_count}/{len(results)} exporters succeeded")
                else:
                    logger.warning(f"Failed to export stats: {results}")
                
            except Exception as e:
                logger.error(f"Error in _dump_stats: {e}", exc_info=True)
        


# Lazy import backend after StrataTelemetry is fully initialized
# This avoids circular import issues
import stratacache.telementry.backend
                
            
