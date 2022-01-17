import os
import torch
import requests
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers.pipelines.automatic_speech_recognition import ffmpeg_read
import numpy as np

# onnx configuration


def get_sample(id=1):
    sample = requests.get(f"https://cdn-media.huggingface.co/speech_samples/sample{id}.flac")
    output_path = Path(f"sample{id}.flac")
    with open(output_path, "wb") as f:
        f.write(sample.content)
    return output_path


def get_inuputs_from_audio(path="sample1.flac", processor=None, tensor_type="pt"):
    with open(path, "rb") as f:
        inputs = ffmpeg_read(f.read(), processor.feature_extractor.sampling_rate)
    return processor(
        inputs,
        return_tensors=tensor_type,
        sampling_rate=processor.feature_extractor.sampling_rate,
    )


def convert_wav2vec2_onnx(
    model_id=None,
    optimize=False,
    quantize=False,
    export_directory="./exports",
    opset_version=13,
    use_gpu=False,
    run_validation=True,
):
    # model that will be converted
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model_name = Path(model_id).stem

    # save path
    os.makedirs(export_directory, exist_ok=True)
    output_path_with_file_name = Path(export_directory).joinpath(f"{model_name}.onnx")
    print(f"save model at: {output_path_with_file_name.absolute()}")

    # dummy input
    sample_path = get_sample(1)
    dummy_model_input = get_inuputs_from_audio(path=sample_path, processor=processor)
    input_names = list(dummy_model_input.keys())
    print("input_names: ", input_names)
    print(
        "current input shape",
        {name: dummy_model_input[name].shape for name in input_names},
    )

    # run pytorch prediction for later comparison and output names
    print(f"Using framework PyTorch: {torch.__version__}")
    with torch.no_grad():
        output = model(*tuple(dummy_model_input.values()))
        output_names = list(output.keys())
        print("output_names: ", output_names)

    # dynamic axes for inputs and outputs
    dynamic_axes_input = {}
    for input_name in input_names:
        dynamic_axes_input[input_name] = {0: "batch_size", 1: "sequence"}

    dynamic_axes_output = {}
    for output_name in output_names:
        dynamic_axes_output[output_name] = {0: "batch_size", 1: "sequence"}
    dynamic_axes = {**dynamic_axes_input, **dynamic_axes_output}
    print("dynamic_axes: ", dynamic_axes)

    # export model
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

    # graph optimization using onnxruntime
    if optimize:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
        import onnx

        opt_output_path = Path(export_directory).joinpath(f"{model_name}.onnx")

        # optimization_options = FusionOptions("bart")

        # TODO: Reenable when fixed: https://github.com/microsoft/onnxruntime/issues/9573
        # optimization_options.enable_embed_layer_norm = False

        # optimized_model = optimizer.optimize_model(
        #     input=output_path_with_file_name.as_posix(),
        #     model_type="bart",
        #     num_heads=model.config.num_attention_heads,
        #     hidden_size=model.config.hidden_size,
        #     use_gpu=True if use_gpu else False,
        #     opt_level=None,  # for now
        #     optimization_options=optimization_options,
        # )
        onnx_model = onnx.load_model(output_path_with_file_name.as_posix())

        optimized_model = optimizer.BartOnnxModel(
            model=onnx_model,
            num_heads=model.config.num_attention_heads,
            hidden_size=model.config.hidden_size,
        )
        optimized_model.optimize()
        print(optimized_model.get_fused_operator_statistics())
        # convert float32 to float 16
        if use_gpu:
            optimized_model.convert_float_to_float16()

        # save model
        optimized_model.save_model_to_file(opt_output_path.as_posix())
        print(f"optimized model saved at: {opt_output_path.absolute()}")

    # quantize model
    if quantize and use_gpu:
        raise ValueError("quantization is not supported for GPU")
    if quantize and not use_gpu:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        model_source_path = opt_output_path if optimize else output_path_with_file_name

        q8_output_path = Path(export_directory).joinpath(f"{model_name}-q8.onnx")

        quantize_dynamic(model_source_path, q8_output_path, weight_type=QuantType.QUInt8)
        print(f"quantized model saved at: {q8_output_path.absolute()}")

    # run checks
    if run_validation:
        import onnxruntime as ort

        # load inference session
        onnx_inputs = get_inuputs_from_audio(path=sample_path, processor=processor, tensor_type="np")
        provider = "CPUExecutionProvider" if not use_gpu else "CUDAExecutionProvider"
        model_path = q8_output_path if quantize else opt_output_path if optimize else output_path_with_file_name

        ort_session = ort.InferenceSession(model_path.as_posix(), providers=[provider])

        onnx_outputs = ort_session.run(None, onnx_inputs.data)[0]
        pytorch_outputs = output[0].cpu().detach().numpy()

        if np.allclose(onnx_outputs, pytorch_outputs, atol=3e-2):
            print("outpus are similar")
        else:
            print("outpus are different")


if __name__ == "__main__":
    model_id = "facebook/wav2vec2-base-960h"
    convert_wav2vec2_onnx(model_id=model_id, optimize=True, quantize=True)
