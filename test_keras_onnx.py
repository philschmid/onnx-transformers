from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

output_path_with_file_name = Path("./exports/keras_model.onnx").absolute()


sample_text = "Hello this is a longer sequence than the model was traced Hello this is a longer sequence than the model was traced Hello this is a longer sequence than the model was traced"
onnx_inputs = tokenizer(sample_text, return_tensors="np")
# change from int64 to int32
onnx_inputs = {k: v.astype("int32") for k, v in onnx_inputs.items()}


ort_session = ort.InferenceSession(
    output_path_with_file_name.as_posix(), providers=["CPUExecutionProvider"]
)
onnx_outputs = ort_session.run(None, onnx_inputs)[0]

print(onnx_outputs[0].shape)
