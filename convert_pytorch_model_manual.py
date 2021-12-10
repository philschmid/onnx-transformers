import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


################# Required variables to convert to onnx ######################

# model that will be converted
model = AutoModel.from_pretrained("distilbert-base-uncased")

# save path
output_path_with_file_name = "./exports/model.onnx"
output_path_with_file_name = Path(output_path_with_file_name).absolute()

# dummy input
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sample_text = "This is a sample sentence."
dummy_model_input = tokenizer(sample_text, return_tensors="pt")

input_names = list(dummy_model_input.keys())
print("input_names: ", input_names)
print("current input shape", {name: dummy_model_input[name].shape for name in input_names})

# output
with torch.no_grad():
    output = model(*tuple(dummy_model_input.values()))
    output_names = list(output.keys())
    print("output_names: ", output_names)

# opset version: List of Operators here https://github.com/onnx/onnx/blob/master/docs/Operators.md
opset_version = 11

# dynamic axes for inputs and outputs
# (dict<string, dict<python:int, string>> or dict<string, list(int)>)
# https://pytorch.org/docs/stable/onnx.html#:~:text=next%20PyTorch%20release%5D-,dynamic_axes,-(dict%3Cstring%2C%20dict
# To specify axes of tensors as dynamic (i.e. known only at run-time), set dynamic_axes to a dict with schema:
# KEY (str): an input or output name. Each name must also be provided in input_names or output_names.
# VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a list, each element is an axis index.
#
# For Transformers we want all input to be dynamic and most of the time all outputs to be dynamic.
# Each input will have as value a dict with the first key being the batch size and the second key being the sequence.

# Not using dynamic axes enables onnx to replace the input ops with a constant for more optimization
# -> there for the dummy input needs to have the correct shape (batch size and sequence length)

dynamic_axes_input = {}
for input_name in input_names:
    dynamic_axes_input[input_name] = {0: "batch_size", 1: "sequence"}

dynamic_axes_output = {}
for output_name in output_names:
    dynamic_axes_output[output_name] = {0: "batch_size", 1: "sequence"}

dynamic_axes = {**dynamic_axes_input, **dynamic_axes_output}
print("dynamic_axes: ", dynamic_axes)
print(f"Using framework PyTorch: {torch.__version__}")

# # Example values:
# input_names:  ['input_ids', 'attention_mask']
# current input shape {'input_ids': torch.Size([1, 8]), 'attention_mask': torch.Size([1, 8])}
# output_names:  ['last_hidden_state']
# dynamic_axes:  {'input_ids': {0: 'batch_size', 1: 'sequence'}, 'attention_mask': {0: 'batch_size', 1: 'sequence'}, 'last_hidden_state': {0: 'batch_size', 1: 'sequence'}}


torch.onnx.export(
    model,  # the model to be exported. (torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction)
    tuple(
        dummy_model_input.values()
    ),  # contain model inputs such that `model(*args)`` is a valid invocation of the model
    f=output_path_with_file_name.as_posix(),  # path to the output file
    input_names=input_names,  # names to assign to the input nodes of the graph, in order.
    output_names=output_names,  # names to assign to the output nodes of the graph, in order.
    dynamic_axes=dynamic_axes,  # To specify axes of tensors as dynamic
    do_constant_folding=True,  # Constant-folding will replace some of the ops that have all constant inputs with pre-computed constant nodes.
    use_external_data_format=False,
    opset_version=opset_version,
)

################# Compare model outputs ######################

import onnxruntime as ort

onnx_inputs = tokenizer(sample_text, return_tensors="np")
print(onnx_inputs)

ort_session = ort.InferenceSession(output_path_with_file_name.as_posix(), providers=["CPUExecutionProvider"])

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
onnx_outputs = ort_session.run(None, onnx_inputs.data)[0]
import numpy as np

pytorch_outputs = output[0].cpu().detach().numpy()

# comapare the outputs for pytorch and onnx
if np.allclose(onnx_outputs, pytorch_outputs, atol=1e-4):
    print("outpus are similar")
