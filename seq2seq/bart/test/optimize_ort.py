import os
from pathlib import Path
from onnxruntime.transformers.fusion_options import FusionOptions

# from onnxruntime.transformers.onnx_model_bart import BartOnnxModel
from bart_model import BartOnnxModel
from onnxruntime.transformers import optimizer
import onnx


num_attention_heads = 24
hidden_size = 768

export_directory = "exports"
model_name = "bart"
model_path = Path("/home/ubuntu/onnx-transformers/seq2seq/bart/test/optimized_BART.onnx")


def main():
    os.makedirs(export_directory, exist_ok=True)
    opt_output_path = Path(export_directory).joinpath(f"{model_name}_opt.onnx")

    # optimization_options = FusionOptions("bart")

    # # TODO: Reenable when fixed: https://github.com/microsoft/onnxruntime/issues/9573
    # optimization_options.enable_embed_layer_norm = False

    # optimized_model = optimizer.optimize_model(
    #     input=model_path.as_posix(),
    #     model_type="bart",
    #     num_heads=12,
    #     hidden_size=768,
    #     opt_level=None,  # for now
    #     optimization_options=optimization_options,
    # )
    onnx_model = onnx.load_model(model_path.as_posix())

    optimized_model = BartOnnxModel(
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
