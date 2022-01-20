# Experiments and examples converting Transformers to ONNX

This repository containes experiments and examples on converting different Transformers to ONNX. 

## Inspect a graph with [Netron](https://github.com/lutzroeder/Netron)

```bash
netron exports/model.onnx
```

netron speech/exports/wav2vec2.onnx


a asdas
 ## References & Resources

* [Transformers documentation](https://huggingface.co/docs/transformers/serialization)
* [Keras/TF to onnx](https://github.com/onnx/tensorflow-onnx)
* [`torch.onnx` export documentation](https://pytorch.org/docs/stable/onnx.html)
* [ONNX Opset documentation with operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
* [export T5 from `fastT5`](https://github.com/Ki6an/fastT5/blob/8dda859086af631a10ad210a5f1afdec64d49616/fastT5/onnx_exporter.py#L45)
* [mrpc example notebook](https://github.com/philschmid/transformers-inference-experiments/blob/main/onnx/simple_mrpc_example.ipynb)
* [Tensorflow Bert example noteboob](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/huggingface-bert.ipynb)
* [Tensorflow2onnx transformers tests](https://github.com/onnx/tensorflow-onnx/blob/master/tests/huggingface.py)
* [Sample Input Tensorflow](https://github.com/onnx/tensorflow-onnx/blob/72d64606d9aff4f93bb064ba901af49377266150/tests/huggingface.py#L129)


# How to optimize the model architecture which isn't yet support in [MODEL_TYPES](https://github.com/microsoft/onnxruntime/blob/001cc539683d5e294d7b306d57e5d5bbb8422d73/onnxruntime/python/tools/transformers/optimizer.py#L36) provided by Microsoft and ONNX

**Requirements:**
* converted `*.onnx` file

Best way to optimize new model architectures, e.g. `wav2vec` is to first look at the architecture and which attention mechanisms are used. In This example wav2vec2 is a copy of the [BartAttention](https://github.com/huggingface/transformers/blob/f00f22a3e290fd377b979124dcf9800b3d73eb11/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L513). 

The first test would be to optimize the model with [BartOnnxModel](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/onnx_model_bart.py) or `model_type` = `bart`, but sadly this did work. 

To understand what we need to change/ what we need to adjust we can copy/fork the closesd "model" in this case the [BartOnnxModel](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/onnx_model_bart.py) and debug it.

## Debugging/creating a custom supported architecture. 

**Step 1:** copy the model file and rename to your architecture. In our example this would mean copy [onnx_model_bart.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/onnx_model_bart.py) and name it `onnx_model_wav2vec2.py`. 

**Step 2:** start netron with the converted model to be able to inspect and search through the graph.

**Step 3:** add a break point and start python debugger. In our example we set a break point in `onnx_model_wav2vec2.py` at line `def create_model(self, config, for_training=True, **kwargs):`
DEBUG with Python debugger and netron. 

1. netron

2. debugger 

3. onnx_model_bart