# Converting Seq2Seq models to ONNX

Currently, it is not easy to convert Seq2Seq models, like BART, T5, Marian to ONNX. So far best practice was to convert the encoder and decoder/s individually and use python to glue it together. 
But it is possible to convert a Seq2Seq model to ONNX using `torch.jit.script` as whole, but for that we need to rewrite some parts of `transformers` to support ONNX.

fatcat-z has created a BART example which can be found in the [transformers repository](https://github.com/huggingface/transformers/tree/master/examples/research_projects/onnx/summarization).


I copied fatcat-z's BART example and used it as base line to create an example for [MarianMT](https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/marian#marianmt). I started with `MarianMT` since it implements a similar Attention mechanism as `BART`.

## Resources

[T5 conversion with individuell files](https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py)
