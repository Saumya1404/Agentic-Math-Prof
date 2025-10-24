import logging
import logging.config
import yaml
import os
from pathlib import Path

def setup_logging(config_path: str | None = None):
    """Load YAML-based logging config."""
    # resolve default config path relative to this file so imports don't depend on CWD
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "logging_config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Logging config not found at {config_path}, using basicConfig")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        return logging.getLogger("app")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ensure directories for file handlers exist (make paths relative to config file)
    handlers = config.get("handlers", {})
    for handler in handlers.values():
        filename = handler.get("filename")
        if filename:
            p = Path(filename)
            if not p.is_absolute():
                p = (config_path.parent / p).resolve()
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                # update the config filename to the resolved absolute path so dictConfig sees it
                handler["filename"] = str(p)
            except Exception:
                # if we can't create dirs, let dictConfig handle/report it
                pass

    try:
        logging.config.dictConfig(config)
    except Exception as e:
        # fallback to console logging so importing the package doesn't crash the process
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger(__name__).exception("dictConfig failed, falling back to basicConfig: %s", e)

    return logging.getLogger("app")


# Initialize a project-wide logger instance
logger = setup_logging()
