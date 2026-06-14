"""
Configuration loader for LiteLLM model gateway.

Description: Loads and applies litellm_config.yaml settings for model
    routing, caching, and fallback configuration.
"""

# Standard library imports
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Third party imports
import yaml
import litellm

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_FILENAME = "litellm_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).parent / CONFIG_FILENAME


# ── Environment Variable Resolution ───────────────────────────────────────────

def _resolve_env_vars(config: Any) -> Any:
    """
    Recursively resolve os.environ/VAR_NAME references in config.

    Description: Walks through config structure and replaces strings
        matching 'os.environ/VAR_NAME' with environment variable values.

    :param config: Configuration value (dict, list, or scalar).
    :return: Configuration with environment variables resolved.
    """
    if isinstance(config, dict):
        return {k: _resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("os.environ/"):
        var_name = config.replace("os.environ/", "")
        value = os.environ.get(var_name)
        if value is None:
            logger.warning(
                f"In function _resolve_env_vars: Environment variable {var_name} not set"
            )
        return value
    return config


# ── Configuration Loader ──────────────────────────────────────────────────────

def load_litellm_config(
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load LiteLLM configuration from YAML file.

    Description: Reads litellm_config.yaml and returns parsed configuration.
        Resolves environment variable references automatically.

    :param config_path: Path to config file. Defaults to package's config.
    :return: Parsed configuration dictionary.
    :raises FileNotFoundError: If config file does not exist.
    :raises yaml.YAMLError: If config file is invalid YAML.
    """
    logger.info("In function load_litellm_config: Entered")

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.error(f"In function load_litellm_config: Config not found at {config_path}")
        raise FileNotFoundError(f"LiteLLM config not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"In function load_litellm_config: Loaded config from {config_path}")
    except yaml.YAMLError:
        logger.exception("In function load_litellm_config: Failed to parse YAML")
        raise

    config = _resolve_env_vars(config)
    logger.info("In function load_litellm_config: Environment variables resolved")

    return config


def apply_litellm_config(
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Apply configuration to LiteLLM library.

    Description: Sets LiteLLM global settings from loaded configuration.
        Call once at application startup.

    :param config: Configuration dict. If None, loads from default path.
    """
    logger.info("In function apply_litellm_config: Entered")

    if config is None:
        config = load_litellm_config()

    # Apply model list
    if "model_list" in config:
        litellm.model_list = config["model_list"]
        logger.info(
            f"In function apply_litellm_config: Registered {len(config['model_list'])} models"
        )

    # Apply litellm settings
    settings = config.get("litellm_settings", {})

    if settings.get("set_verbose"):
        litellm.set_verbose = True

    if settings.get("drop_params"):
        litellm.drop_params = True
        logger.info("In function apply_litellm_config: Enabled drop_params")

    if "success_callback" in settings:
        litellm.success_callback = settings["success_callback"]

    if "failure_callback" in settings:
        litellm.failure_callback = settings["failure_callback"]

    logger.info("In function apply_litellm_config: Configuration applied")


def get_model_config(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific model tier.

    Description: Looks up model configuration by name (local, mid, cloud).

    :param model_name: Name of the model tier.
    :param config: Configuration dict. If None, loads from default path.
    :return: Model configuration dict, or None if not found.
    """
    logger.info(f"In function get_model_config: Entered for model={model_name}")

    if config is None:
        config = load_litellm_config()

    for model in config.get("model_list", []):
        if model.get("model_name") == model_name:
            logger.info(f"In function get_model_config: Found config for {model_name}")
            return model

    logger.warning(f"In function get_model_config: No config found for {model_name}")
    return None
