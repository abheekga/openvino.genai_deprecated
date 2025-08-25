import logging as log
import sys


    # from optimum.intel.openvino import OVDiffusionPipeline

#from optimum.intel.openvino import OVStableDiffusionXLPipeline
import argparse
import os
from transformers import AutoProcessor
from optimum.intel.openvino import OVWeightQuantizationConfig, OVModelForCausalLM
from pathlib import Path
import os

import shutil
import nncf
import openvino as ov
import gc
import glob
import venv

def export_llava_next_video(ov_model_path):
    model_id = "llava-hf/llava-next-video-7B-hf"
    MODEL_DIR = ov_model_path

    if not (MODEL_DIR / "FP16").exists():
        os.system(f"optimum-cli export openvino --model llava-hf/LLaVA-NeXT-Video-7B-hf --weight-format fp16 {MODEL_DIR}/FP16")

    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 64,
        "ratio": 0.6,
    }

    core = ov.Core()
    LANGUAGE_MODEL_PATH_INT4 = MODEL_DIR / "INT4/openvino_language_model.xml"
    LANGUAGE_MODEL_PATH = MODEL_DIR / "FP16/openvino_language_model.xml"
    if not LANGUAGE_MODEL_PATH_INT4.exists():
        ov_model = core.read_model(LANGUAGE_MODEL_PATH)
        ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
        ov.save_model(ov_compressed_model, LANGUAGE_MODEL_PATH_INT4)
        del ov_compressed_model
        del ov_model
        gc.collect()

        copy_model_folder(MODEL_DIR / "FP16", MODEL_DIR / "INT4", ["openvino_language_model.xml", "openvino_language_model.bin"])


def copy_model_folder(src, dst, ignore_file_names=None):
    ignore_file_names = ignore_file_names or []

    for file_name in Path(src).glob("*"):
        if file_name.name in ignore_file_names:
            continue
        shutil.copy(file_name, dst / file_name.relative_to(src))


def export_model_with_optimum(ov_model_path, model_id, weight="int4"):
    if not os.path.exists(ov_model_path):
        os.system(f"optimum-cli export openvino --trust-remote-code --model {model_id} --weight-format {weight} {ov_model_path}")

    return 0

def export_gemma():
    model_id = "google/gemma-3-4b-it"
    quantization_config = OVWeightQuantizationConfig(bits=4, sym=False)

    model = OVModelForCausalLM.from_pretrained(model_id, device="GPU", quantization_config=quantization_config, trust_remote_code=True)
    model.save_pretrained("./models/gemma-3-4b-it")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.save_pretrained("./models/gemma-3-4b-it")

def run_model_with_benchmark(input, output, ov_model_path, prompt_in):
    print(f"Input Size: {input}, Output Size: {output}")

    prompt = f"prompts/{prompt_in}.jsonl"
    
    os.system(f"python benchmark.py -m {ov_model_path} -d GPU -n 2 -ic {output} -pf {prompt}")
    
    return 0

def run_model_with_vlm_benchmark(input, output, ov_model_path, height, width):
    print(f"Input Size: {input}, Output Size: {output}")

    prompt = "Sometimes it's nice to take a minute in the pew by yourself beforehand. You have this beautiful church probably almost all to yourself. Can you feel its energy resonating through you? Can you feel the majesty of the Lord's kingdom and how you're a part of it? Take a moment to kneel and pray with your head down and hands clasped together. Think about how you've been responding to God's call and how you've been living in the light of his love."

    image = f"images/image_{height}_{width}.jpg"
    
    os.system(f'python ../../samples/python/visual_language_chat/benchmark_vlm.py -m {ov_model_path} -d GPU -mt {output} -i {image} -p "{prompt}"')
    
    return 0

def clear_storage_space():
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

    total, used, free = shutil.disk_usage("C:\\")
    threshold = 25 * (1024**3)
    if free < threshold:
        print(f"Free space on C drive is below 25 GB, currently at ({free / (1024**3):.2f}) GB remaining.")
        # clear_storage_space()
    else:
        print(f"Free space on C drive is above 25 GB, currently at ({free / (1024**3):.2f}) GB remaining.")

    if args.model=="gemma3":
        ov_model_path = "models/gemma-3-4b-it"
        model_id = "google/gemma-3-4b-it"
        if not os.path.exists(ov_model_path):
            export_gemma()
        run_model_with_benchmark(args.input, args.output, ov_model_path, "100")
    elif args.model=="minicpm-v":
        ov_model_path = "models/MiniCPM-V-2_6"
        model_id = "openbmb/MiniCPM-V-2_6"
        export_model_with_optimum(ov_model_path, model_id)
        run_model_with_vlm_benchmark(args.input, args.output, ov_model_path, args.height, args.width)
    elif args.model=="llava-llama":
        ov_model_path = "models/llava3-llama-next"
        model_id = "llava-hf/llama3-llava-next-8b-hf"
        export_model_with_optimum(ov_model_path, model_id)
        run_model_with_vlm_benchmark(args.input, args.output, ov_model_path, args.height, args.width)
    elif args.model=="phi3.5-vision":
        ov_model_path = "models/phi3.5-vision"
        model_id = "microsoft/Phi-3.5-vision-instruct"
        export_model_with_optimum(ov_model_path, model_id)
        run_model_with_vlm_benchmark(args.input, args.output, ov_model_path, args.height, args.width)
    elif args.model=="phi4-vision":
        ov_model_path = "models/phi4-multimodal-instruct"
        model_id = "microsoft/Phi-4-multimodal-instruct"
        export_model_with_optimum(ov_model_path, model_id)
        run_model_with_vlm_benchmark(args.input, args.output, ov_model_path, args.height, args.width)
    elif args.model=="llava-video":
        ov_model_path = "models/llava-next-video-7B-ov/INT4"
        if not os.path.exists(ov_model_path):
            export_llava_next_video(ov_model_path)
        run_model_with_benchmark(args.input, args.output, ov_model_path, "100_llava")
    else:
        raise(ValueError("Unsupported pipeline"))

    return 0

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--input", default=1024)
    parser.add_argument("--output", default=128)
    parser.add_argument("--height", default=512)
    parser.add_argument("--width", default=512)
    args=parser.parse_args()
    main(args)
