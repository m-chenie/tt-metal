# LLM Provider Configuration

The kernel generator supports multiple LLM providers.

## Supported Providers

### Groq (Default)
- Fast inference using LPU chips
- Free tier available
- Best model: `llama-3.3-70b-versatile`

```bash
export GROQ_API_KEY="your-groq-api-key"
python3 generate_kernel.py --operation diode_equation --core-mode single --generate-host
```

### OpenAI
- GPT-4 family models
- Best for complex reasoning and code generation
- Recommended model: `gpt-4o-2024-08-06`

```bash
export OPENAI_API_KEY="your-openai-api-key"
python3 generate_kernel.py --provider openai --operation diode_equation --core-mode single --generate-host
```

## Usage Examples

### Generate with Groq (default)
```bash
python3 generate_kernel.py --operation diode_equation --core-mode single --generate-host
```

### Generate with OpenAI GPT-4o
```bash
python3 generate_kernel.py --provider openai --operation diode_equation --core-mode single --generate-host
```

### Specify a custom model
```bash
# Groq with specific model
python3 generate_kernel.py --provider groq --model llama-3.1-70b-versatile --operation diode_equation --core-mode single

# OpenAI with GPT-4 Turbo
python3 generate_kernel.py --provider openai --model gpt-4-turbo --operation diode_equation --core-mode single
```

### Iterative refinement with OpenAI
```bash
python3 generate_kernel.py --provider openai --iterate --example-path /path/to/example --max-iterations 5 --save-prompt
```

## Model Recommendations

### For Speed (Groq)
- `llama-3.3-70b-versatile` - Best balance of speed and quality
- `llama-3.1-8b-instant` - Fastest, lower quality

### For Quality (OpenAI)
- `gpt-4o-2024-08-06` - Best overall reasoning and code generation
- `gpt-4-turbo` - Good balance of cost and quality
- `gpt-4` - Most accurate but slower

## Cost Comparison

**Groq:** Free tier available, very fast
- llama-3.3-70b: Free (with limits)

**OpenAI:** Pay per token
- gpt-4o: ~$2.50-$10 per million tokens (input/output)
- gpt-4-turbo: ~$10-$30 per million tokens

For kernel generation with ~20K input + 6K output tokens:
- Groq: Free
- OpenAI GPT-4o: ~$0.06 per generation

**Note:** Better models often require fewer iterations, potentially saving costs overall.
