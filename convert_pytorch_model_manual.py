import torch
from transformers import AutoModel, AutoTokenizer
from transformers.convert_graph_to_onnx import convert

################# Required variables to convert to onnx ######################

# model that will be converted
model = AutoModel.from_pretrained("distilbert-base-uncased")

# save path
output_path_with_file_name = "./model.onnx"

# dummy input
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sample_text = "This is a sample sentence."
tokens = tokenizer(sample_text, return_tensors="pt")

# output
output = model(**tokens)

print(f"Using framework PyTorch: {torch.__version__}")

# with torch.no_grad():
#     input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "pt")
#     ordered_input_names, model_args = ensure_valid_input(nlp.model, tokens, input_names)

# torch.export.onnx(
#     model,  # the model to be exported. (torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction)
#     model_args,  # contain model inputs such that `model(*args)`` is a valid invocation of the model
#     f=output.as_posix(),
#     input_names=ordered_input_names,
#     output_names=output_names,
#     dynamic_axes=dynamic_axes,
#     do_constant_folding=True,
#     use_external_data_format=use_external_format,
#     enable_onnx_checker=True,
#     opset_version=opset,
# )

# torch.export.onnx(
#     model=task.model,
#     args=(dict(encodings),),
#     f=output_path.as_posix(),
#     input_names=list(config.task.inputs.keys()),
#     output_names=list(config.task.outputs.keys()),
#     opset_version=self.default_opset,
#     dynamic_axes=dict(
#         **{name: dict(v) for name, v in dynamic_inputs_axes.items()},
#         **{name: dict(v) for name, v in dynamic_outputs_axes.items()},
#     ),
#     use_external_data_format=use_external_data_format,
# )
