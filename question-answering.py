import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
import json
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from typing import Dict, Optional, Sequence


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="squad-llama-2-7B/checkpoint-10950")


@dataclass
class InferenceArguments:
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
      )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
    )
    inference_dtype: torch.dtype = field(
        # default=torch.float32,
        default=torch.bfloat16,
        metadata={"help": "The dtype to use for inference."},
    )


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.cuda()
    model.eval()

    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # use_fast=False,
        model_max_length=inference_args.model_max_length,
    )

    with open("squad/Preprocessed_dev-v1.1.json", "r") as f:
        questions = json.load(f)
    
    results = []
    for question in questions:
        ctx = question["instruction"] + "\n" + question["input"] + "\n### Response:"
        # ctx = question["instruction"] + " " + question["input"] + "\n### Response:"
        inputs = tokenizer(ctx, return_tensors="pt")
        outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
            max_new_tokens=50,
            # generation_config=generation_config,
            output_scores=False,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded = decoded[decoded.rfind("### Response:") + 13:].strip()
        print("Answer: " + decoded)
        print("------------------------------------------")
        results.append({"instruction": question["instruction"], "input": question["input"], "output": question["output"], "response": decoded})
    
    print("writing to squad-answers/llama-2-7B-c10950.json")
    with open("squad-answers/llama-2-7B-c10950.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    inference()