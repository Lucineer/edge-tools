# Smart Model Router Skill — Route to Cheapest Capable Model

Automatically picks the cheapest AI model that can handle a given task. Reduces costs up to 70%.

## Usage

```
router/smart <prompt> [preferred-model]
router/budget <max-cost-per-M-tokens>
router/estimate <prompt> <model>
```

## Examples

```
# Best model for money, accounting, or spreadsheets
router/smart "calculate the compound interest on $5000 at 8% over 10 years"
# → routes to deepseek-reasoner (best at math, $0.55/$2.19 per M)

# Simple chat routed to cheapest
router/smart "hello, how are you?"
# → routes to deepseek-chat ($0.14/$0.28 per M)

# Stay under budget
router/budget 0.60
# → selects cheapest model under $0.60/1M output

# Estimate cost before sending
router/estimate "explain quantum computing" deepseek-chat
# → $0.000042 estimated
```

## Routing Logic

1. Classifies message into task type (math, code, analysis, research, vision, simple chat)
2. Scores each model on: capability match + cost + speed preference
3. Picks highest-scoring model
4. Falls back to cheapest if nothing matches

## Cost Savings

| Pattern | Default (GPT-4o) | Routed | Savings |
|---------|------------------|--------|---------|
| Simple chat | $2.50/$10.00 | $0.14/$0.28 | 94-97% |
| Code review | $2.50/$10.00 | $0.80/$4.00 | 60-68% |
| Math problem | $2.50/$10.00 | $0.55/$2.19 | 78-82% |

## Implementation

- `router-config.js` in cocapn-chat repo
- 8 regex patterns matching task types
- 8 models with capability tags
- Budget-aware fallback
- Speed preference support
