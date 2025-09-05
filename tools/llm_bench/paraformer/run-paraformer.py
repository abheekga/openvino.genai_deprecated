import requests

r = requests.get(
    # url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/b393399955cded1779e5eaf7be4349bc2109c4be/utils/notebook_utils.py",
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py", "w").write(r.text)

model_id = "model"
wav_path = "distil-whisper-librispeech-long_30s.wav"


from funasr import AutoModel
from funasr_onnx import Paraformer
import openvino as ov
from notebook_utils import device_widget
from timeit import Timer
from funasr_onnx.utils.postprocess_utils import sentence_postprocess, sentence_postprocess_sentencepiece
import os
from huggingface_hub import snapshot_download


local_dir = "model"
if not os.path.exists(local_dir):
    new_dir = snapshot_download(repo_id="abheekga/paraformer-zh", local_dir=local_dir)
onnx_model = Paraformer(model_id, batch_size=1, quantize=False)

wav_path = [wav_path]

# result = onnx_model(wav_path)
# print(result)
# exit()

waveform_list = onnx_model.load_data(wav_path, onnx_model.frontend.opts.frame_opts.samp_freq)

feats, feats_len = onnx_model.extract_feat(waveform_list[0:1])

print(feats.shape, feats.dtype, feats_len, feats_len.dtype)

device = 'GPU'

onnx_model_path = model_id + "/model.onnx"
ov_model_path = model_id + "/ov_model.xml"
print(onnx_model_path)

core = ov.Core()

if not os.path.exists(ov_model_path):
    compiled_model_onnx = core.compile_model(model=onnx_model_path, device_name=device)

    ov_model = ov.convert_model(onnx_model_path, example_input=dict({"speech": feats, "speech_lengths": feats_len}))
    ov.save_model(ov_model, model_id + "/ov_model.xml")


ov_model = core.read_model(ov_model_path)
compiled_model = core.compile_model(ov_model, device)

res_lambda = lambda: compiled_model(dict({"speech": feats, "speech_lengths": feats_len}))
reps = Timer(res_lambda).repeat(repeat=1+5, number=1)
print(reps)
print(min(reps))