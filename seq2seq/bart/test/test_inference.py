import onnxruntime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import time

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

test_input = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
inputs = tokenizer(test_input, return_tensors="pt")

num_beams = 5
max_length = 50
decoder_start_token_id = 2

ort_sess = onnxruntime.InferenceSession("optimized_BART.onnx")
st = time.time()
ort_out = ort_sess.run(
    None,
    {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy(),
        "num_beams": np.array(num_beams),
        "max_length": np.array(max_length),
        "decoder_start_token_id": np.array(decoder_start_token_id),
    },
)
print(f"ORT: inference for num_beams:{num_beams} & max_length:{max_length}  took {time.time() - st} seconds")
print(f"Prediction Text:\n")
print(tokenizer.decode(ort_out[0][0], skip_special_tokens=True))

model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

st = time.time()
summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    num_beams=num_beams,
    max_length=max_length,
    early_stopping=True,
    decoder_start_token_id=2,
)
print(f"Pytorch: inference for num_beams:{num_beams} & max_length:{max_length}  took {time.time() - st} seconds")
print(f"Prediction Text:\n")
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
