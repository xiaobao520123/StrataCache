"""
Configuration management for StrataCache.

This module provides easy access to configuration values from config.yaml.
Supports hierarchical config with getters, defaults, and type checking.
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional, Union, Dict
import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for StrataCache."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader.
        
        Args:
            config_path: Path to config.yaml. If None, searches in default locations:
                        - ./config.yaml (current directory)
                        - ../config.yaml (parent directory)
                        - ~/.stratacache/config.yaml (user home)
        """
        self._config = {}
        self._config_path = config_path or self._find_config_file()
        
        if self._config_path and os.path.exists(self._config_path):
            self._load_yaml()
        else:
            logger.warning(f"Config file not found: {self._config_path}")
    
    def _find_config_file(self) -> Optional[str]:
        """Find config.yaml in default locations."""
        locations = [
            "./config.yaml",
            "../config.yaml",
            os.path.expanduser("~/.stratacache/config.yaml"),
            os.path.join(os.path.dirname(__file__), "config.yaml"),
        ]
        
        for loc in locations:
            if os.path.exists(loc):
                logger.info(f"Found config file: {loc}")
                return loc
        
        return None
    
    def _load_yaml(self):
        """Load YAML configuration file."""
        try:
            with open(self._config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {self._config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {self._config_path}: {e}")
            self._config = {}
    
    def reload(self):
        """Reload configuration from file."""
        self._load_yaml()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-separated key path.
        
        Args:
            key: Config key path (e.g., 'stratacache.connector.use_cxl')
            default: Default value if key not found
            
        Returns:
            Config value or default
            
        Example:
            config = Config()
            use_cxl = config.get('stratacache.connector.use_cxl', False)
        """
        parts = key.split('.')
        value = self._config
        
        try:
            for part in parts:
                if isinstance(value, dict):
                    value = value[part]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get config value as integer."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Config value {key}={value} cannot be converted to int, using default {default}")
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get config value as float."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Config value {key}={value} cannot be converted to float, using default {default}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get config value as boolean."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return bool(value)
    
    def get_str(self, key: str, default: str = "") -> str:
        """Get config value as string."""
        value = self.get(key, default)
        if value is None:
            return default
        return str(value)
    
    def get_list(self, key: str, default: Optional[list] = None) -> list:
        """Get config value as list."""
        if default is None:
            default = []
        value = self.get(key, default)
        if isinstance(value, list):
            return value
        if value is None or value == "":
            return default
        return [value]
    
    def get_dict(self, key: str, default: Optional[dict] = None) -> dict:
        """Get config value as dictionary."""
        if default is None:
            default = {}
        value = self.get(key, default)
        if isinstance(value, dict):
            return value
        return default
    
    def set(self, key: str, value: Any):
        """Set config value by dot-separated key path.
        
        Args:
            key: Config key path (e.g., 'stratacache.connector.use_cxl')
            value: Value to set
            
        Example:
            config = Config()
            config.set('stratacache.connector.use_cxl', True)
        """
        parts = key.split('.')
        current = self._config
        
        # Create nested dicts if needed
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                logger.warning(f"Cannot set {key}: {part} is not a dict")
                return
            current = current[part]
        
        # Set the final value
        current[parts[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save config to YAML file.
        
        Args:
            path: Path to save config. If None, saves to original location.
        """
        save_path = path or self._config_path
        if not save_path:
            logger.error("No path specified and original config path unknown")
            return
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            logger.info(f"Saved config to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get entire config as dictionary."""
        return dict(self._config)
    
    def __repr__(self) -> str:
        return f"Config(path={self._config_path})"


# Global singleton instance
_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create the global config instance.
    
    Args:
        config_path: Optional path to config file (only used on first call)
        
    Returns:
        Config instance
        
    Example:
        config = get_config()
        use_cxl = config.get_bool('stratacache.connector.use_cxl')
    """
    global _instance
    if _instance is None:
        _instance = Config(config_path)
    return _instance


def reset_config():
    """Reset the global config instance (mainly for testing)."""
    global _instance
    _instance = None


# Convenience functions for common config values
def get_connector_config() -> Dict[str, Any]:
    """Get connector configuration section."""
    return get_config().get_dict('stratacache.connector', {})


def get_wandb_enabled() -> bool:
    """Check if WanDB is enabled in config."""
    return get_config().get_bool('stratacache.connector.wandb.enabled', False)

def get_wandb_entity() -> Optional[str]:
    """Get WanDB entity from config."""
    return get_config().get_str('stratacache.connector.wandb.entity', None)

def get_wandb_project() -> Optional[str]:
    """Get WanDB project from config."""
    return get_config().get_str('stratacache.connector.wandb.project', "stratacache")

def get_wandb_run_name() -> Optional[str]:
    """Get WanDB run name from config."""
    return get_config().get_str('stratacache.connector.wandb.run_name', "default-run")


def get_cpu_capacity_mb() -> int:
    """Get CPU capacity in MB."""
    return get_config().get_int('stratacache.connector.cpu_capacity_mb', 61440)


def get_chunk_size() -> int:
    """Get chunk size configuration."""
    return get_config().get_int('stratacache.connector.chunk_size', 256)


def get_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return get_config().get_bool('stratacache.connector.debug', False)


def is_cxl_enabled() -> bool:
    """Check if CXL is enabled."""
    return get_config().get_bool('stratacache.connector.use_cxl', False)
