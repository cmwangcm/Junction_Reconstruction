"""USD per 1M tokens for the OpenRouter demo provider.

Inner key: exact model id string used in API calls.
Unmatched model still reports tokens, with cost = null.
"""

PRICING = {
    "openrouter": {
        "openai/gpt-5.6-sol": {"input": 5.0, "output": 30.0, "cache_read": 0.50, "cache_write": 0.0},
    },
}
