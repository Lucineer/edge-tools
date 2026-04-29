# Edge Router Skill — Jetson On-Device Model Routing

Select the best local AI model for a given task on the Jetson Orin Nano.

## Usage

```
edge/router <prompt> [quality]
edge/router-models
edge/router-benchmark <model-name>
```

## Examples

```
# Route to best model for code generation
edge/router "write a quicksort in rust"

# Route with quality preference
edge/router "explain quantum computing" low

# List available edge models
edge/router-models

# Run benchmark
edge/router-benchmark DeepSeek-R1-Distill-Qwen-1.5B
```

## How It Works

1. Classifies task into: chat, code, reasoning, writing, embedding
2. Filters available models that fit on hardware (8GB RAM, 1024 CUDA cores)
3. Selects based on task type + quality preference
4. For code/reasoning: prefers larger models (slower but more capable)
5. For chat/writing: prefers faster models

## Available Models

| Model | Size | Runtime | Speed |
|-------|------|---------|-------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.2GB | LiteRT-LM | 35 t/s |
| Qwen2.5-0.5B | 0.4GB | LiteRT-LM | 60 t/s |
| SmolLM-135M | 0.1GB | LiteRT-LM | 120 t/s |

## Notes

- Runs entirely on-device, no API calls needed
- All models are quantized LiteRT-LM compatible
- Available RAM after model load: ~6.8GB with DeepSeek-1.5B
