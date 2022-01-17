from numpy.lib.twodim_base import vander
import torch
from pathlib import Path
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import tf2onnx


import warnings

warnings.filterwarnings("ignore")


################# Required variables to convert to onnx ######################

# model that will be converted
model = TFAutoModel.from_pretrained("distilbert-base-uncased")

# save path
output_path_with_file_name = "./exports/keras_model.onnx"
output_path_with_file_name = Path(output_path_with_file_name).absolute()

# dummy input
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sample_text = "This is a sample sentence."
dummy_model_input = tokenizer(sample_text, return_tensors="tf")


# print("current input shape", {name: dummy_model_input[name].shape for name in input_names})

# output
output = model(dummy_model_input)
print(output)

# opset version: List of Operators here https://github.com/onnx/onnx/blob/master/docs/Operators.md
opset_version = 11

print(f"Using framework Tensorflow: {tf.__version__} \ntf2onnx: {tf2onnx.__version__}")

# create tensorflow input: nice implemented function here: https://github.com/onnx/tensorflow-onnx/blob/72d64606d9aff4f93bb064ba901af49377266150/tests/huggingface.py#L129
tf_input_spec = []
for k, v in dummy_model_input.items():
    tf_input_spec.append(tf.TensorSpec((None, None), v.dtype, name=k))
tf_input_spec = tuple(tf_input_spec)
print("tf_input_spec: ", tf_input_spec)


tf2onnx.convert.from_keras(
    model,
    input_signature=tf_input_spec,
    opset=opset_version,
    output_path=output_path_with_file_name,
)


# ################# Compare model outputs ######################

import onnxruntime as ort

onnx_inputs = tokenizer(sample_text, return_tensors="np")
# change from int64 to int32
onnx_inputs = {k: v.astype("int32") for k, v in onnx_inputs.items()}


ort_session = ort.InferenceSession(
    output_path_with_file_name.as_posix(), providers=["CPUExecutionProvider"]
)

print("Inputs:")
for idx, inputs in enumerate(ort_session.get_inputs()):
    print("idx:", idx)
    print("   Name:", inputs.name)
    print("   Shape:", inputs.shape)
print("Outputs:")
for idx, outputs in enumerate(ort_session.get_outputs()):
    print("idx:", idx)
    print("   Name:", outputs.name)
    print("   Shape:", outputs.shape)


# onnx_inputs is a userdict data is stored a .data
onnx_outputs = ort_session.run(None, onnx_inputs)[0]
import numpy as np

tf_outputs = output[0].numpy()

# comapare the outputs for pytorch and onnx
if np.allclose(onnx_outputs, tf_outputs, atol=1e-4):
    print("outpus are similar")
