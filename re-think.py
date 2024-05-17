import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
import json
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from typing import Dict, Optional, Sequence
import string
import re


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="squad-llama-2-7B")


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


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_answer_exactly_match(origin_answer, right_answer):
    alternative_answers = []
    if "output" in right_answer:
        alternative_answers.append(normalize_answer(right_answer["output"]))
    if "output0" in right_answer:
        alternative_answers.append(normalize_answer(right_answer["output0"]))
    if "output1" in right_answer:
        alternative_answers.append(normalize_answer(right_answer["output1"]))
    if "output2" in right_answer:
        alternative_answers.append(normalize_answer(right_answer["output2"]))
    if "output3" in right_answer:
        alternative_answers.append(normalize_answer(right_answer["output3"]))
    if "output4" in right_answer:
        alternative_answers.append(normalize_answer(right_answer["output4"]))
    if "output5" in right_answer:
        alternative_answers.append(normalize_answer(right_answer["output5"]))
    return normalize_answer(origin_answer["response"]) in alternative_answers


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

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=inference_args.model_max_length,
    )

    with open("squad-answers/llama-2-7B.json", "r") as f:
        origin_answers = json.load(f)
    with open("squad/Preprocessed_dev-v1.1.json", "r") as f:
        right_answers = json.load(f)
    
    new_results = []
    for i in range(len(origin_answers)):
        origin_answer = origin_answers[i]
        right_answer = right_answers[i]
        is_right = is_answer_exactly_match(origin_answer, right_answer)
        if is_right:
            new_results.append(origin_answer)
            print(origin_answer["response"] + " is right. Pass.")
        else:
            ctx = origin_answer["instruction"] + "\n" + origin_answer["input"] + "\n" + origin_answer["response"] + "is the wrong answer, please think again.\n### Response:"
            inputs = tokenizer(ctx, return_tensors="pt")
            outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                max_new_tokens=50,
                # generation_config=generation_config,
                output_scores=False,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded = decoded[decoded.rfind("### Response:") + 13:].strip()
            print("New Answer: " + decoded)
            origin_answer["response"] = decoded
            new_results.append(origin_answer)
        print("---------------------------------------")
    
    print("writing to squad-answers/llama-2-7B-rethink-v2.json")
    with open("squad-answers/llama-2-7B-rethink-v2.json", "w") as f:
        json.dump(new_results, f, indent=4, ensure_ascii=False)

    # ctx = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.\n" + \
    #     "What city did Super Bowl 50 take place in?\n" + "San Francisco Bay Area is the wrong answer, please think again.\n### Response:"
    # inputs = tokenizer(ctx, return_tensors="pt")
    # outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
    #     max_new_tokens=50,
    #     # generation_config=generation_config,
    #     output_scores=False,
    # )
    # decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # decoded = decoded[decoded.rfind("### Response:") + 13:].strip()
    # print("Answer: " + decoded)

    # with open("squad/Preprocessed_dev-v1.1.json", "r") as f:
    #     questions = json.load(f)
    
    # results = []
    # for question in questions:
    #     ctx = question["instruction"] + "\n" + question["input"] + "\n### Response:"
    #     # ctx = question["instruction"] + " " + question["input"] + "\n### Response:"
    #     inputs = tokenizer(ctx, return_tensors="pt")
    #     outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
    #         max_new_tokens=50,
    #         generation_config=generation_config,
    #         output_scores=False,
    #     )
    #     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     decoded = decoded[decoded.rfind("### Response:") + 13:].strip()
    #     print("Answer: " + decoded)
    #     print("------------------------------------------")
    #     results.append({"instruction": question["instruction"], "input": question["input"], "output": question["output"], "response": decoded})
    
    # print("writing to squad-answers/llama-2-7B-c10950.json")
    # with open("squad-answers/llama-2-7B-c10950.json", "w") as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    inference()