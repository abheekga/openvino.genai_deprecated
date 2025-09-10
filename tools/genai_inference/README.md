# Python genai sample

This example showases inference of several categories of models including large language models and visual language models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the weight format on the export to int4 or fp16.

The code given uses the openvino.genai repo to run the following two benchmarks:
 - [`benchmark.py`](../llm_bench/benchmark.py) demonstrates basic usage of the LLM pipeline.
 - [`benchmark_vlm.py`](../../samples/python/visual_language_chat/benchmark_vlm.py) shows how to benchmark a VLM in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text and calculating various performance metrics.

## Run benchmark:

```sh
python run_genai_inference.py [OPTIONS]
```

### Options

- `-m, --model`(default: `None`): Path to the model and tokenizers base directory.
- `-c, --category` (default: `None`): The category of models, either choose from vlm or llm.`
- `--input` (default: `1024`): Size of input prompt file, can choose from sizes included in prompts directory under ./tools/llm_bench/prompts.
- `--output` (default: `128`): Output token count, maximum tokens generated will not exceed this number.
- `--height` (default: `512`): Image resolution height. For visual language models only.
- `--width` (default: `512`): Image resolution width. For visual language models only.
- `--config` (default: `False`): Loading PA backend with updated openvino.genai. Temporarily supported for visual language models.

### Output:

```
python run_genai_inference.py -c vlm -m minicpm-v
```

```
res= The sky appears blue because of a natural phenomenon called Rayleigh scattering. When sunlight reaches the Earth's atmosphere, it encounters molecules like nitrogen and oxygen. These molecules scatter the shorter, blue wavelengths of light more than the longer, red wavelengths. This scattering makes the sky appear blue, especially during the day when the sun is shining brightly.
Load time: 7822.00 ms
Generate time: 832.31 ± 4.91 ms
Tokenization time: 2.06 ± 0.14 ms
Detokenization time: 0.29 ± 0.01 ms
Embeddings preparation time: 65.93 ± 0.00 ms
TTFT: 106.16 ± 1.58 ms
TPOT: 10.83 ± 2.30 ms/token 
Throughput: 92.32 ± 19.64 tokens/s
```

### List of Supported Models

```
LLM Models Supported:
llama2 (7b, int4)
llama3.2 (3b, int4)
llama3.1 (8b, int4)
GLM-Edge (4b, int4)
Qwen2.5-7B chat (int4)
Qwen3-0.6B base (FP16)
Qwen3-8B base (int4)
Phi-3.5-mini(3.8b, int4)
Phi4 mini reasoning 3.8B  (int4)
Phi4 mini instruct 3.8B (int4)
Gemma1 (7b, int4)
Mistral 7b (int4)
MiniCPM1-1B (FP16)
DeepSeek R1 (Qwen 14b, int4)
```

```
VLM Models Supported:
gemma 3 4b (int4)
miniCPM-V2.6 (Qwen2-7B, int4)
miniCPM-o2.6 (Qwen2-7B, int4)
llava3-next-llama3 (Llama-3-8B-Instruct, int4)
LLaVA-NeXT-Video-7B-hf(int4)
phi3.5 vision (fp16)
phi4 vision (fp16)
```
