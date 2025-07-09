# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from datasets import load_dataset
import os
import copy
import random
from tqdm import tqdm
try:
    from verl.utils.tool_utils import _parse_function_string, _extract_functions_from_system, _validate_function_format
except:
    from utils.tool_utils import _parse_function_string, _extract_functions_from_system, _validate_function_format
import random
import copy
from hammer_utils import replace_param_names_new, replace_function_names_new, replace_param_default_values_news

DEFAULT_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. \nBased on the question, you will need to make one or more function/tool calls to achieve the purpose. \nIf none of the function can be used, point it out. If the given question lacks the parameters required by the function,\nalso point it out. You should only return the function call in tools call sections. Here is a list of functions in JSON format that you can invoke:
"""

def _process_toolace(data, include_nocall=False):

    filtered = []
    for d in data:
        tools = _extract_functions_from_system(d["system"])
        if not tools:
            continue
        item = {"tools": json.dumps(tools)}
        marker = "JSON format that you can invoke:\n"
        idx = d["system"].find(marker)
        item["system"] = d["system"][: idx + len(marker)]
        convs = []
        for c in d["conversations"]:
            if c["from"] == "gpt" and c["value"].startswith("[") and c["value"].endswith("]"):
                funcs = _parse_function_string(c["value"])
                if funcs and _validate_function_format(c["value"]):
                    c["from"] = "function_call"
                    c["value"] = json.dumps([{"name": f[0], "arguments": f[1]} for f in funcs])
            convs.append(c)
        item["conversations"] = convs
        filtered.append(item)

    results = []
    for d in filtered:
        cs, sys, tools = d["conversations"], d["system"], d["tools"]
        for i, c in enumerate(cs):
            if c["from"] == "function_call" and c["value"].startswith("[") and c["value"].endswith("]"):
                results.append({"system": sys, "conversations": cs[:i], "answer": json.loads(c["value"]), "tools": tools})
            if c["from"] == "gpt" and include_nocall:
                results.append({"system": sys, "conversations": cs[:i], "answer": [], "tools": tools})
    
    results = [r for r in results if r["tools"]]
    final = []
    for r in results:
        if len(r["conversations"]) >=2:
            id = "toolace_multiple_turn"
        else:
            id = "toolace_single_turn"
        out = {
            "raw_system": r["system"],
            "tools": json.loads(r["tools"]),
            "conversations": r["conversations"],
            "answer": r["answer"],
            "id": id
        }
        for c in out["conversations"]:
            if c["from"] == "function_call" and c["value"].startswith("[") and c["value"].endswith("]"):
                c["from"] = "function_call"
        final.append(out)

    return [
        f for f in final
        if all(ans["name"] in {tool["name"] for tool in f["tools"]} for ans in f["answer"])
    ]


def _preprocess_xlam(origin_data):

    processed_data = []
    tools = origin_data["tools"]
    answers = origin_data["answers"]
    queries = origin_data["query"]
    length = len(origin_data["id"])
    print("length", length)
    for tem in range(length):
        processed_tem = {}
        processed_tem["tools"] = tools[tem]
        processed_tem["conversations"] = []
        processed_tem["conversations"].append({"from": "human", "value": queries[tem]})
        functions = json.loads(answers[tem])
        for fun in functions:
            processed_tem["conversations"].append({"from": "function_call", "value": json.dumps(fun)})
        processed_data.append(processed_tem)

    final_data = []
    for data in processed_data:
        data["tools"] = json.loads(data["tools"])
        conversation = {"from": "human", "value": data["conversations"][0]["value"]}
        answer = []
        for index, item in enumerate(data["conversations"]):
            if item["from"] == "function_call":
                answer.append(json.loads(item["value"]))
        data["answer"] = answer
        data["conversations"] = [conversation]
        data["raw_system"] = DEFAULT_PROMPT
        data["id"] = "xlam"
        final_data.append(data) 

    return final_data

def pre_process_data(data_names, include_nocall=False):

    # toolace - single turn
    if data_names == "toolace_single_turn":
        with open("path/to/your/data/toolace_single_turn.json", 'r') as f:
            dataset_dict = json.load(f)
        print("raw_toolace_single_turn", len(dataset_dict))
        result = _process_toolace(dataset_dict)

    # toolace - multiple turn
    elif data_names == "toolace_multi_turn":
        with open("path/to/your/data/toolace_multi_turn.json", 'r') as f:
            dataset_dict = json.load(f)
        print("raw_toolace_multi_turn", len(dataset_dict))
        result = _process_toolace(dataset_dict, include_nocall=include_nocall)

    # xlam - single turn 
    elif data_names == "xlam":
        dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train", cache_dir="data/raw/xlam/")
        dataset_dict = dataset.to_dict()
        result = _preprocess_xlam(dataset_dict)
    else:
        raise ValueError(f"Error: The dataset '{data_names}' is not supported. Please check the dataset name or implement support for it.")
    
    return result

def data_augment_irrelevant(raw_data, sample_rate = 0.15):
    data_copy = copy.deepcopy(raw_data)
    final_result = []
    sample_size = int(sample_rate * len(data_copy))
    used_indices = set()
    for _ in range(sample_size):
        while True:
            if len(used_indices) >= len(data_copy):
                raise ValueError("no enough samples.")
            
            idx = random.randrange(len(data_copy))

            if idx in used_indices or data_copy[idx]["id"] not in ["xlam"]:
                continue

            item = copy.deepcopy(data_copy[idx])

            answer_tool_names = [tool["name"] for tool in item["answer"]]
            item["tools"] = [tool for tool in item["tools"] if tool["name"] not in answer_tool_names]
            
            if item["tools"] and len(item["tools"]) > 0:
                item["answer"] = []
                final_result.append(item)
                used_indices.add(idx)
                break
            else:
                continue

    return final_result


def hammer_mask(raw_data, mask_raio=0.67):

    data_mask = []
    for it in tqdm(raw_data):
        if it["id"] != "xlam":
            raise ValueError("Error: The dataset is not supported. Please check the dataset name or implement support for it.")
        mask_threshold = 1-mask_raio
        if random.random()> mask_threshold:
            it["tools"] = json.dumps(it["tools"])
            it["answers"] = json.dumps(it["answer"])
            it["query"] = it["conversations"][0]["value"]
            it = replace_param_default_values_news(replace_param_names_new(replace_function_names_new(it)))
            it["tools"] = json.loads(it["tools"])
            it["answer"] = json.loads(it["answers"])
            it["conversations"][0]["value"] = it["query"]
            del it["query"]
            del it["answers"]
        data_mask.append(it)

    return data_mask


def main():

    parser = argparse.ArgumentParser(description="Process a dataset from Hugging Face")
    parser.add_argument("--output_dir", default="data/", help="Output JSON file")
    args = parser.parse_args()

    data_sum = []

    multiple_turn_toolace = pre_process_data("toolace_multi_turn")
    print("toolace_multi_turn", len(multiple_turn_toolace))

    data_sum.extend(multiple_turn_toolace)

    single_turn_toolace = pre_process_data("toolace_single_turn")
    print("toolace_single_turn", len(single_turn_toolace))
    data_sum.extend(single_turn_toolace)
    xlam = pre_process_data("xlam")
    print("xlam", len(xlam))
    data_sum.extend(xlam) 
    save_path = os.path.join(args.output_dir,"raw_data.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_sum, f, ensure_ascii=False, indent=4)
    print(f"Processed data saved to {save_path}")

if __name__ == "__main__":
    main()