from pathlib import Path
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnx_model_wav2vec2 import Wav2vec2OnnxModel
import onnx


num_attention_heads = 12
hidden_size = 768

export_directory = "exports"
model_name = "wav2vec2"
model_path = Path("speech/exports/wav2vec2-base-960h.onnx")


def main():

    opt_output_path = Path(export_directory).joinpath(f"{model_name}_opt.onnx")

    onnx_model = onnx.load_model(model_path.as_posix())

    optimized_model = Wav2vec2OnnxModel(
        model=onnx_model,
        num_heads=num_attention_heads,
        hidden_size=hidden_size,
    )

    # Will call https://github.com/microsoft/onnxruntime/blob/e5ee0b435db9007921adeadffe929f247a5d6055/onnxruntime/python/tools/transformers/onnx_model_bert.py#L302
    optimized_model.optimize()
    print(optimized_model.get_fused_operator_statistics())

    # save model
    optimized_model.save_model_to_file(opt_output_path.as_posix())
    print(f"optimized model saved at: {opt_output_path.absolute()}")


if __name__ == "__main__":
    main()
