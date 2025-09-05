from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoTokenizer
import openvino_genai
import librosa
import os
import time

output_dir = "models/whisper-large-v3"

if not os.path.exists(output_dir):
    model = OVModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3", export=True, trust_remote_code=True)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large-v3")
    export_tokenizer(tokenizer, output_dir)


def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

pipe = openvino_genai.WhisperPipeline(output_dir, "GPU")
# Pipeline expects normalized audio with Sample Rate of 16kHz
raw_speech = read_wav('30_second_test_out.wav')

start = time.perf_counter()
result = pipe.generate(raw_speech)
end = time.perf_counter()

total_time = (end - start) * 1000.0
print(f"Total time to generate is {total_time} ms")

