import os

# Local proxy for the OpenRouter demo. Set to None for a direct connection.
DEFAULT_PROXY = "http://127.0.0.1:2022"

API_PROVIDERS = {
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "proxy": DEFAULT_PROXY,
    },
}


def resolve_provider(name):
    if name not in API_PROVIDERS:
        raise KeyError("unknown API provider: %r (choose from %s)" % (name, ", ".join(sorted(API_PROVIDERS))))
    cfg = API_PROVIDERS[name]
    env_name = cfg["api_key_env"]
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise RuntimeError("API key for %r is not set; export %s" % (name, env_name))
    return {
        "source": name,
        "api_key": api_key,
        "base_url": cfg["base_url"],
        "proxy": cfg.get("proxy"),
    }
