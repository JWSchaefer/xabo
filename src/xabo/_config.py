import tomllib


def load_config() -> dict:
    try:
        with open('pyproject.toml', 'rb') as f:
            config = tomllib.load(f)
            config = config.get('tool', {}).get('xabo', {})
            return config
    except Exception:
        return {}
