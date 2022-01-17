from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

output_path_with_file_name = Path("./exports/qa_model.onnx").absolute()

# manuall pipeline

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
onnx_inputs = tokenizer(question, text, return_tensors="np")
# change from int64 to int32

ort_session = ort.InferenceSession(
    output_path_with_file_name.as_posix(), providers=["CPUExecutionProvider"]
)
onnx_outputs = ort_session.run(None, onnx_inputs.data)[:2]

start_scores = np.argmax(onnx_outputs[0])
end_scores = np.argmax(onnx_outputs[1])

answer_ids = onnx_inputs["input_ids"][0][start_scores : end_scores + 1]
answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids, skip_special_tokens=True)
print([{"answer": tokenizer.convert_tokens_to_string(answer_tokens)}])


# pipeline
class OnnxModel:
    def __init__(self, model=None, config=None, **kwargs):
        self.model = model
        self.config = config

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_all_providers,
)


class OnnxModel:
    def __init__(self, model_path: str, provider: str):
        assert (
            provider in get_all_providers()
        ), f"provider {provider} not found, {get_all_providers()}"
        self.options = self._set_options()
        self.model = InferenceSession(model_path, self.options, providers=[provider])
        # Load the model as a graph and prepare the CPU backend
        self.model.disable_fallback()

    def __call__(self, input):
        return self.model.run(None, input)

    def _set_options(self):
        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        return options


model = OnnxModel(
    output_path_with_file_name.as_posix(), provider="CPUExecutionProvider"
)
print(model(onnx_inputs.data))

from transformers import QuestionAnsweringPipeline, Pipeline

# https://github.com/huggingface/transformers/blob/b53bc55ba9bb10d5ee279eab51a2f0acc5af2a6b/src/transformers/pipelines/base.py#L87
# ONNX Pipelines won't work Out of the Box since there are Hard checks for Pytorch and Tensorflow
# Option would be to add a ONNXPreTrainedModel
# https://colab.research.google.com/drive/1kt5WdZnzqn9K9gYkX4EHlNlM4vdvasrh#scrollTo=je8ingd0m9rQ


class OptimumQuestionAnsweringPipeline(QuestionAnsweringPipeline, Pipeline):
    def __init__(self, **kwargs):
        Pipeline.__init__(self, **kwargs)

    def forward(self, model_inputs, **forward_params):
        print(model_inputs)
        print(forward_params)
        onnx_outputs = self.model(model_inputs)
        return QuestionAnsweringModelOutput(
            start_logits=onnx_outputs[0], end_logits=onnx_outputs[1]
        )


qa = OptimumQuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
print(qa(question, text))
# output =
# print(output)
