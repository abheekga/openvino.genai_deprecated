import logging as log
import sys
import argparse
import os
import shutil
import glob
import psutil
import subprocess
import datetime
import threading
import time

class ModelInfo:
    def __init__(self, ov_model_path, weight, model_id, task):
        self.ov_model_path = ov_model_path
        self.weight = weight
        self.model_id = model_id
        self.task = task

# Global model configuration map
MODEL_MAP = {
    "llama2": ModelInfo("models/llama-2-7b", "int4", "meta-llama/Llama-2-7b-chat-hf", False),
    "llama3.2": ModelInfo("models/llama-3.2-3b", "int4", "meta-llama/Llama-3.2-3B", False),
    "llama3.1": ModelInfo("models/llama-3.1-8b", "int4", "meta-llama/Llama-3.1-8B", False),
    "glm": ModelInfo("models/glm-edge-4b", "int4", "zai-org/glm-edge-4b-chat", False),
    "qwen2.5": ModelInfo("models/Qwen2.5-7B-Instruct", "int4", "Qwen/Qwen2.5-7B-Instruct", False),
    "qwen3-0.6": ModelInfo("models/Qwen3-0.6B", "fp16", "Qwen/Qwen3-0.6B", False),
    "qwen3-8": ModelInfo("models/Qwen3-8B", "int4", "Qwen/Qwen3-8B", False),
    "phi-3.5": ModelInfo("models/Phi-3.5-mini-instruct", "int4", "microsoft/Phi-3.5-mini-instruct", False),
    "phi4-reason": ModelInfo("models/Phi-4-mini-reasoning", "int4", "microsoft/Phi-4-mini-reasoning", False),
    "phi4-instruct": ModelInfo("models/Phi-4-mini-instruct", "int4", "microsoft/Phi-4-mini-instruct", False),
    "gemma1": ModelInfo("models/gemma1-7b", "int4", "google/gemma-7b", False),
    "mistral": ModelInfo("models/Mistral-7B-Instruct", "int4", "mistralai/Mistral-7B-Instruct-v0.3", True),
    "minicpm": ModelInfo("models/Minicpm-1b-sft-bf16", "fp16", "openbmb/MiniCPM-1B-sft-bf16", False),
    "deepseek": ModelInfo("models/Deepseek-R1-Distill-Qwen-14B", "int4", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", False),
}

def measure_memory(output_file_name, stop_event, compile_event):
    with open(output_file_name, mode='w') as output_file:
        while not stop_event.is_set():
            gpu_mem_cmd = r'(((Get-Counter "\GPU Process Memory(*)\Local Usage").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'

            def run_command(command):
                val = subprocess.run(['powershell', '-Command', command], capture_output=True).stdout.decode("ascii")
                return float(val.strip().replace(',', '.')) / 2**20

            gpu_memory = run_command(gpu_mem_cmd)
            cpu_memory = (psutil.virtual_memory().total - psutil.virtual_memory().available) / 1024 / 1024
            now = datetime.datetime.now()
            output_file.write(f"Timestamp: {now.time()}\n")
            output_file.write(f"GPU Memory Usage: {gpu_memory:.2f} MB\n")
            output_file.write(f"CPU Memory Usage: {cpu_memory:.2f} MB\n")
            if compile_event.is_set():
                output_file.write("Compilation finished.\n")
                compile_event.clear()
            output_file.flush()  # Ensure data is written immediately

def run_model(input, output, model_info, mem=False):

    logger = log.getLogger()

    logger.info(f"Input Size: {input}, Output Size: {output}")

    prompt = f"../prompts/{input}_tokens_test.jsonl"
    if not os.path.exists(prompt):
        logger.error(f"Prompt file for {input} tokens does not exist")
        raise(ValueError(f"Prompt file for {input} tokens does not exist"))
    
    if not os.path.exists(model_info.ov_model_path):
        os.system(f"huggingface-cli download {model_info.model_id} --local-dir {model_info.ov_model_path}")
        if not os.path.exists(model_info.ov_model_path):
            logger.error(f"Model path {model_info.ov_model_path} does not exist after download")
            raise(ValueError(f"Model path {model_info.ov_model_path} does not exist after download"))
        # Remove gguf file to save disk space; we're rebuilding it anyway
        if model_info.model_id == "google/gemma-7b":
            os.remove(os.path.join(model_info.ov_model_path, "gemma-7b.gguf"))

    gguf_file = os.path.join(os.getcwd(), "model.gguf")
    quant_file = os.path.join(os.getcwd(), "quant.gguf")

    if mem:
        output_file = f"memory_log_{model_info.model_id.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        stop_event = threading.Event()
        compile_event = threading.Event()
        logging_thread = threading.Thread(target=measure_memory, args=(output_file, stop_event, compile_event))
        logging_thread.start()
        time.sleep(2)  # Ensure logging thread starts before benchmark
        logger.info("Memory logging started.")
        os.system(f"python ./llama.cpp/convert_hf_to_gguf.py {model_info.ov_model_path} --outtype auto --outfile {gguf_file}")
        logger.info("Model conversion to GGUF completed.")
        os.system(f".\\llama-vulkan\\llama-quantize.exe {gguf_file} {quant_file} Q4_K_M")
        logger.info("Model quantization to Q4_K_M completed.")
        compile_event.set()
        time.sleep(4)  # Give some time for the logging thread to log the compilation event
        os.system(f".\\llama-vulkan\\llama-cli.exe -m {quant_file} -f \"{prompt}\" -n {output} --simple-io -st")
        logger.info("Inference completed.")
        time.sleep(2)  # Give it some time to collect the idle memory just in case
        stop_event.set()
        logging_thread.join()
        logger.info("Memory logging stopped.")

    else:
        os.system(f"python ./llama.cpp/convert_hf_to_gguf.py {model_info.ov_model_path} --outtype auto --outfile {gguf_file}")
        logger.info("Model conversion to GGUF completed.")
        os.system(f".\\llama-vulkan\\llama-quantize.exe {gguf_file} {quant_file} Q4_K_M")
        logger.info("Model quantization to Q4_K_M completed.")
        os.system(f".\\llama-vulkan\\llama-cli.exe -m {quant_file} -f \"{prompt}\" -n {output} --simple-io -st")
        logger.info("Inference completed.")
    
    os.remove(gguf_file)
    os.remove(quant_file)
    logger.info("Temporary model files removed.")

    return 0

def clear_storage_space():
    # Clears the cache and model directory
    CACHE_DIR = "C:/Users/gta/.cache/huggingface/hub"
    MODEL_DIR = "./models"
    
    TARGET_DIRS = [CACHE_DIR, MODEL_DIR]
    print(f"Clearing {CACHE_DIR} and {MODEL_DIR}")

    for target in TARGET_DIRS:
        abs_path = os.path.abspath(target)
        if os.path.exists(abs_path):
            shutil.rmtree(abs_path, ignore_errors=True)
        else:
            print(f"Directory not found: {abs_path}")

    # Files may be temporarily added to temp app data, clearing only the openvino files
    # from that directory
    TEMP_DIR = os.path.join(os.environ["LOCALAPPDATA"], "Temp")
    
    pattern = os.path.join(TEMP_DIR, "**", "*.bin")

    for file_path in glob.iglob(pattern, recursive=True):
        if "openvino" in file_path.lower():
            try:
                print(f"Deleting {file_path}")
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def main(args):
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    if not args.model and not args.all:
        raise ValueError("Either --model or --all must be specified")
    models_to_run = MODEL_MAP.keys() if args.all else [args.model]
    for model in models_to_run:
        if model not in MODEL_MAP:
            raise ValueError(f"Unsupported model: {model}")
        log.info(f"Running model: {model}")
        total, used, free = shutil.disk_usage("C:\\")
        threshold = 50 * (1024**3)
        if free < threshold:
            print(f"Free space on C drive is below 50 GB, currently at ({free / (1024**3):.2f}) GB remaining.")
            # User can clear storage space if needed.
            clear_storage_space()
        else:
            print(f"Free space on C drive is above 50 GB, currently at ({free / (1024**3):.2f}) GB remaining.")
        
        model_info = MODEL_MAP[model]

        run_model(args.input, args.output, model_info, mem=args.mem)

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m")
    parser.add_argument("--input", default=1024)
    parser.add_argument("--output", default=128)
    parser.add_argument("--mem", default=False, action="store_true")
    parser.add_argument("--all", default=False, action="store_true", help="Run all models sequentially")
    args=parser.parse_args()
    main(args)
