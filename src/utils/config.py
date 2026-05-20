import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal environments
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only in minimal environments
    load_dotenv = None


def _load_env_file(env_path: Path) -> None:
    """Load .env without requiring python-dotenv during Sprint 0 smoke tests."""
    if load_dotenv is not None:
        load_dotenv(env_path)
        return

    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _strip_inline_comment(value: str) -> str:
    """Strip YAML-style comments outside quotes."""
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[:index].rstrip()
    return value.strip()


def _parse_scalar(value: str):
    value = _strip_inline_comment(value)
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> dict:
    """
    Minimal two-level YAML parser for configs/config.yaml.

    It keeps Sprint 0 runnable before dependencies are installed. Full YAML files
    should still use PyYAML, which is listed in requirements.txt.
    """
    config = {}
    current_section = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if indent == 0:
            if value:
                config[key] = _parse_scalar(value)
                current_section = None
            else:
                config[key] = {}
                current_section = key
        elif current_section:
            config[current_section][key] = _parse_scalar(value)

    return config


_load_env_file(PROJECT_ROOT / ".env")

def get_project_root() -> Path:
    """Return the absolute path of the project root."""
    return PROJECT_ROOT

def load_config() -> dict:
    """Load configuration from configs/config.yaml and resolve relative paths to absolute."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read()

    if yaml is not None:
        config = yaml.safe_load(config_text)
    else:
        config = _parse_simple_yaml(config_text)

    # Resolve paths relative to PROJECT_ROOT
    if "paths" in config:
        for key, val in config["paths"].items():
            config["paths"][key] = str(PROJECT_ROOT / val)

    return config

def get_env_var(name: str, default: str = "") -> str:
    """Get an environment variable value."""
    return os.getenv(name, default)
