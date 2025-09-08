#!/usr/bin/env python
# coding: utf-8

# # Omnimodal assistant with MiniCPM-o 2.6 and OpenVINO
# 
# MiniCPM-o 2.6 is the latest and most capable model in the MiniCPM-o series. The model is built in an end-to-end fashion based on SigLip-400M, Whisper-medium-300M, ChatTTS-200M, and Qwen2.5-7B with a total of 8B parameters. It exhibits a significant performance improvement over MiniCPM-V 2.6, and introduces new features for real-time speech conversation and multimodal live streaming.
# 
# More details about model can be found in [model card](https://huggingface.co/openbmb/MiniCPM-o-2_6) and original [repo](https://github.com/OpenBMB/MiniCPM-O).
# 
# In this tutorial we consider how to convert and optimize MiniCPM-o 2.6 model for creating omnimodal chatbot. Additionally, we demonstrate how to apply stateful transformation on LLM part and model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf)
# 
# #### Table of contents:
# 
# - [Prerequisites](#Prerequisites)
# - [Convert model to OpenVINO Intermediate Representation](#Convert-model-to-OpenVINO-Intermediate-Representation)
#     - [Compress Language Model Weights to 4 bits](#Compress-Language-Model-Weights-to-4-bits)
# - [Prepare model inference pipeline](#Prepare-model-inference-pipeline)
#     - [Select device](#Select-device)
#     - [Select language model variant](#Select-language-model-variant)
# - [Run OpenVINO model inference](#Run-OpenVINO-model-inference)
#     - [Omni mode](#Omni-mode)
#     - [Vision-Only mode](#Vision-Only-mode)
# - [Interactive demo](#Interactive-demo)
# 
# 
# ### Installation Instructions
# 
# This is a self-contained example that relies solely on its own code.
# 
# We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
# For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).
# 
# <img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm-o-omnimodal-chatbot/minicpm-o-omnimodal-chatbot.ipynb" />
# 

# ## Prerequisites
# [back to top ⬆️](#Table-of-contents:)

# In[1]:


# get_ipython().run_line_magic('pip', 'install -q "torch>=2.3.1" "torchvision>=0.18.1" "librosa>=0.9.0" "vocos>=0.1.0"  "vector-quantize-pytorch>=1.18.5" "torchaudio>=2.3.1" # "soundfile>=0.12.1" "timm>=0.9.2" "transformers==4.44.2" "Pillow" "gradio>=4.19"  "tqdm" "sentencepiece" "peft" "huggingface-hub>=0.24.0" "moviepy>=1.0.3" --no-cache-dir -- # extra-index-url https://download.pytorch.org/whl/cpu')
# get_ipython().run_line_magic('pip', 'install -q "openvino>=2025.1.0" "nncf>=2.16.0"')
# get_ipython().run_line_magic('pip', 'install -q --force-reinstall numpy==2.2.0')


# In[1]:


import requests
from pathlib import Path

if not Path("minicpm_o_helper.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/minicpm-o-multimodal-chatbot/minicpm_o_helper.py"
    )
    open("minicpm_o_helper.py", "w").write(r.text)


if not Path("gradio_helper.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/minicpm-o-multimodal-chatbot/gradio_helper.py")
    open("gradio_helper.py", "w").write(r.text)

if not Path("notebook_utils.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
    open("notebook_utils.py", "w").write(r.text)

# Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
from notebook_utils import collect_telemetry

collect_telemetry("minicpm-o-omnimodal-chatbot.ipynb")


# ## Convert model to OpenVINO Intermediate Representation
# [back to top ⬆️](#Table-of-contents:)
# 
# OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate Representation (IR). [OpenVINO model conversion API](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model) should be used for these purposes. `ov.convert_model` function accepts original PyTorch model instance and example input for tracing and returns `ov.Model` representing this model in OpenVINO framework. Converted model can be used for saving on disk using `ov.save_model` function or directly loading on device using `core.complie_model`.
# 
# `minicpm_helper.py` script contains helper function for model conversion, please check its content if you interested in conversion details.
# 
# <details>
#   <summary><b>Click here for more detailed explanation of conversion steps</b></summary>
# MiniCPM-o 2.6 is autoregressive transformer generative model, it means that each next model step depends from model output from previous step. The generation approach is based on the assumption that the probability distribution of a word sequence can be decomposed into the product of conditional next word distributions. In other words, model predicts the next token in the loop guided by previously generated tokens until the stop-condition will be not reached (generated sequence of maximum length or end of string token obtained). The way the next token will be selected over predicted probabilities is driven by the selected decoding methodology. You can find more information about the most popular decoding methods in this [blog](https://huggingface.co/blog/how-to-generate). The entry point for the generation process for models from the Hugging Face Transformers library is the `generate` method. You can find more information about its parameters and configuration in the [documentation](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate). To preserve flexibility in the selection decoding methodology, we will convert only model inference for one step.
# 
# The inference flow has difference on first step and for the next. On the first step, model accept preprocessed input instruction, audio and image, that transformed to the unified embedding space using `text embedding`, `audio encoder` and `image encoder` models, after that `language model`, LLM-based part of model, runs on input embeddings to predict probability of next generated tokens. 
# 
# With increasing model size like in modern LLMs, we also can note an increase in the number of attention blocks and size past key values tensors respectively. The strategy for handling cache state as model inputs and outputs in the inference cycle may become a bottleneck for memory-bounded systems, especially with processing long input sequences, for example in a chatbot scenario. OpenVINO suggests a transformation that removes inputs and corresponding outputs with cache tensors from the model keeping cache handling logic inside the model. Such models are also called stateful. A stateful model is a model that implicitly preserves data between two consecutive inference calls. The tensors saved from one run are kept in an internal memory buffer called a `state` or a `variable` and may be passed to the next run, while never being exposed as model output. Hiding the cache enables storing and updating the cache values in a more device-friendly representation. It helps to reduce memory consumption and additionally optimize model performance. More details about stateful models and working with state can be found in [OpenVINO documentation](https://docs.openvino.ai/2024/openvino-workflow/running-inference/stateful-models.html).
# 
# In LLMs, `text_embedding` is a part of language model, but for multimodal case, the first step hidden state produced by this model part should be integrated with image embeddings into common embedding space. For ability to reuse this model part and avoid introduction of llm model instance, we will use it separately.
# 
# To sum up above, model consists of 4 parts:
# 
# * **Image Encoder** for encoding input images into embedding space. It includes SigLIP model.
# * **Audio Encoder** for encoding input audio into embedding space. It includes Whisper model.
# * **Resampler** for compression image representation.
# * **Text Embedding** for conversion input text tokens into embedding space.
# * **Language Model** for generation answer based on input embeddings provided by Image Encoder and Input Embedding models.
# 
# Let's convert each model part.
# </details>

# In[2]:


from minicpm_o_helper import convert_minicpmo26

model_id = "openbmb/MiniCPM-o-2_6"

model_dir = convert_minicpmo26(model_id)


# ### Compress Language Model Weights to 4 bits
# [back to top ⬆️](#Table-of-contents:)
# 
# For reducing memory consumption, weights compression optimization can be applied using [NNCF](https://github.com/openvinotoolkit/nncf). 
# 
# <details>
#     <summary><b>Click here for more details about weight compression</b></summary>
# Weight compression aims to reduce the memory footprint of a model. It can also lead to significant performance improvement for large memory-bound models, such as Large Language Models (LLMs). LLMs and other models, which require extensive memory to store the weights during inference, can benefit from weight compression in the following ways:
# 
# * enabling the inference of exceptionally large models that cannot be accommodated in the memory of the device;
# 
# * improving the inference performance of the models by reducing the latency of the memory access when computing the operations with weights, for example, Linear layers.
# 
# [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) provides 4-bit / 8-bit mixed weight quantization as a compression method primarily designed to optimize LLMs. The main difference between weights compression and full model quantization (post-training quantization) is that activations remain floating-point in the case of weights compression which leads to a better accuracy. Weight compression for LLMs provides a solid inference performance improvement which is on par with the performance of the full model quantization. In addition, weight compression is data-free and does not require a calibration dataset, making it easy to use.
# 
# `nncf.compress_weights` function can be used for performing weights compression. The function accepts an OpenVINO model and other compression parameters. Compared to INT8 compression, INT4 compression improves performance even more, but introduces a minor drop in prediction quality.
# 
# More details about weights compression, can be found in [OpenVINO documentation](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html).
# 
# </details>
# 
# > **Note:** weights compression process may require additional time and memory for performing. You can disable it using widget below:

# In[3]:


# from minicpm_o_helper import compression_widget

# to_compress_weights = compression_widget()

# to_compress_weights


# In[4]:


import nncf
import gc
import openvino as ov

from minicpm_o_helper import llm_path, copy_llm_files


compression_configuration = {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 1.0, "all_layers": True}


core = ov.Core()
llm_int4_path = Path("language_model_int4") / llm_path.name
if not (model_dir / llm_int4_path).exists():
    ov_model = core.read_model(model_dir / llm_path)
    ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
    ov.save_model(ov_compressed_model, model_dir / llm_int4_path)
    del ov_compressed_model
    del ov_model
    gc.collect()
    copy_llm_files(model_dir / llm_path.parent, model_dir / llm_int4_path.parent)


# ## Prepare model inference pipeline
# [back to top ⬆️](#Table-of-contents:)
# 
# As discussed, the model comprises Image Encoder and LLM (with separated text embedding part) that generates answer. In `minicpm_helper.py` we defined LLM inference class `OvModelForCausalLMWithEmb` that will represent generation cycle, It is based on [HuggingFace Transformers `GenerationMixin`](https://huggingface.co/docs/transformers/main_classes/text_generation) and looks similar to [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) `OVModelForCausalLM`that is used for LLM inference with only difference that it can accept input embedding. In own turn, general multimodal model class `OvMiniCPMVModel` handles chatbot functionality including image processing and answer generation using LLM.

# ### Select device
# [back to top ⬆️](#Table-of-contents:)

# In[5]:


# from notebook_utils import device_widget

# device = device_widget(default="AUTO", exclude=["NPU"])

device = "GPU"


# ### Select language model variant
# [back to top ⬆️](#Table-of-contents:)

# In[6]:


from minicpm_o_helper import init_model


# use_int4_lang_model = lm_variant_selector(model_dir / llm_int4_path)

# use_int4_lang_model


# In[7]:


ov_model = init_model(model_dir, llm_int4_path.parent, device)
tokenizer = ov_model.processor.tokenizer


# ## Run model inference
# [back to top ⬆️](#Table-of-contents:)
# 
# Let's explore model capabilities using multimodal input

# ### Omni mode
# [back to top ⬆️](#Table-of-contents:)
# 
# This is an example for real-time video understanding, omni-source (video & audio) understanding, and multimodal contextual understanding.

# In[9]:

print("Video and Audio mode omni")
print("VIDEO AND AUDIO INPUT, TEXT OUTPUT")

import math
import numpy as np
from PIL import Image

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip
import tempfile
import librosa


def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print("video_duration:", video.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = "sample_temp_demo.wav"
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

    # 1 frame + 1s audio chunk
    contents = []
    for i in range(num_units):
        frame = video.get_frame(i + 1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr * i : sr * (i + 1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])

    return contents


# video_path = "MiniCPM-o-2_6/ckpt/assets/Skiing.mp4"
video_path = "../prompts/SampleVideo_1280x720_1mb.mp4"
# if use voice clone prompt, please set ref_audio
ref_audio_path = "MiniCPM-o-2_6/ckpt/assets/demo.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
sys_msg = ov_model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")
# or use default prompt
# sys_msg = model.get_sys_prompt(mode='omni', language='en')

contents = get_video_chunk_content(video_path)
question = "What is in the image? Please make sure to give an extensive description and go on for as long as possible."
msg = {"role": "user", "content": [contents, question]}
# msgs = [sys_msg, msg]
msgs = [msg]

res = ov_model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.5,
    max_new_tokens=128,
    # omni_input=True,  # please set omni_input=True when omni inference
    use_tts_template=True,
    max_slice_nums=1,
    use_image_id=False,
    return_dict=True,
)
print("Token generated: ")
print(res)


# ### Vision-Only mode
# [back to top ⬆️](#Table-of-contents:)
# 
# In Vision-Only mode, MiniCPM-o-2_6 has the same inference methods as MiniCPM-V-2_6.

# In[10]:

print("Vision only Mode: ")
print("IMAGE INPUT, TEXT OUTPUT")
import requests
from io import BytesIO

image_path = Path("cat.png")

if not image_path.exists():
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB").resize((512, 512))
    image.save(image_path)
else:
    image = Image.open(image_path).convert("RGB").resize((512, 512))
question = "What is in the image? Please make sure to give an extensive description and go on for as long as possible."
msgs = [{"role": "user", "content": [image, question]}]
## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = ov_model.chat(msgs=msgs, tokenizer=tokenizer, sampling=True)
generated_text = ""
print("Token generated: ")
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end="")


