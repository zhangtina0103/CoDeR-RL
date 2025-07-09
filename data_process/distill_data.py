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

from openai import OpenAI
import json, os
import time
import re
from collections import Counter

ds_client = OpenAI(api_key="", base_url="https://api.deepseek.com")

def validate_result(result, answer):
    if len(result) == 0 or len(answer) == 0:
        if len(result) == len(answer):
            return 2
        else:
            return 0
    else:
        try:
            counter1_full = Counter((item["name"], json.dumps(item["arguments"], sort_keys=True)) 
                                    for item in result)
            counter2_full = Counter((item["name"], json.dumps(item["arguments"], sort_keys=True)) 
                                    for item in answer)
        except TypeError:
            return 0
        if counter1_full == counter2_full:
            return 2
        
        counter1_name = Counter(item["name"] for item in result)
        counter2_name = Counter(item["name"] for item in answer)

        if counter1_name == counter2_name:
            return 1
        
        return 0

def validate_format(tool_call_list):
    for item in tool_call_list:
        if not isinstance(item, dict):
            return 0
    for item in tool_call_list:
        if "name" not in item.keys() or "arguments" not in item.keys():
            return 0
    return 1

def extract_output(tool_call_str):

    marker = "<|eot_id|>"
    index = tool_call_str.find(marker)
    if index != -1:
        tool_call_str = tool_call_str[:index]
        
    output_string = tool_call_str

    return output_string

def extract_solution(tool_call_str):
    
    output_string = extract_output(tool_call_str)

    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = list(re.finditer(pattern, tool_call_str, flags=re.DOTALL))
    if not matches:
        return None, output_string
    last_content = matches[-1].group(1).strip()
    try:
        return json.loads(last_content),output_string
    except json.JSONDecodeError:
        return None, output_string

def format_tools(tools):

    string = ""
    for tool in tools:
        string += json.dumps({"type": "function", "function": tool}) + "\n"
    if string[-1] == "\n":
        string = string[:-1]
    return string
    
def get_r1_response(prompt_input, tools,patience=10):

    tools_string = format_tools(tools)

    system_prompt =f"""
    You are an expert in composing functions. You are given a question and a set of possible functions. \nBased on the question, you will need to make one or more function/tool calls to achieve the purpose. \nIf none of the function can be used, point it out. If the given question lacks the parameters required by the function,\nalso point it out. You should only return the function call in tools call sections.\n You are provided with function signatures within <tools></tools> XML tags. Here is a list of functions in JSON format that you can invoke:\n

    <tools>
    {tools_string}
    </tools>

    In each action step, you MUST: 
    1. Think about the reasoning process in the mind and enclosed your reasoning within <think> </think> XML tags.
    2. Then, provide a json object with function names and arguments within <tool_call></tool_call> XML tags. i.e., <tool_call>[{{"name": <function-name>, "arguments": <args-json-object>}}, {{"name": <function-name2>, "arguments": <args-json-object2>}}, ...]</tool_call>
    3. Make sure both the reasoning and the tool call steps are included together in one single reply.
    A complete reply example is: <think>To address the query, I need to send the email to Bob and then buy the banana through walmart. </think> <tool_call> [{{"name": "email", "arguments": {{"receiver": "Bob", "content": "I will bug banana through walmart"}}}}, {{"name": "walmart", "arguments": {{"input": "banana"}}}}]</tool_call>. Please make sure the type of the arguments is correct."""

    query_prompt = f"""{prompt_input}"""

    while patience > 0:
        patience -= 1
        try:
            completion = ds_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt},
                ],
                stream=False
            )
            return True, completion.choices[0].message.content, system_prompt

        except Exception as e:
            print(e)
            time.sleep(5)

    return False, '', system_prompt

def check_result(all_string, result, answer):
    if result != None:
        if "<think>" not in all_string or "</think>" not in all_string:
            return 0
    if result is None:
        return 0
    if not validate_format(result):
        return 0
    if validate_result(result, answer) == 2:
        return 1
    else:
        return 0

if __name__ == "__main__":

    save_path = "path/to/your/data.json"
    index_path = save_path + ".idx"
    with open('path/to/your/raw_data', 'r') as file:
        raw_dataset = json.load(file)
    tool_ace_data = [item for item in raw_dataset if item["id"] == "toolace_single_turn"]
    start_idx = 0
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            start_idx = int(f.read().strip())

    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            json.dump([], f)

    with open(save_path, 'r') as f:
        saved_data = json.load(f)

    for idx, data_instance in enumerate(tool_ace_data):
        if idx < start_idx:
            continue
        answer = data_instance["answer"]
        tools = data_instance["tools"]
        query = data_instance["conversations"][0]["value"]
        failed = 0
        success, output, system_prompt = get_r1_response(query, tools)
        if success:
            output_string = extract_output(output)
            output_extracted, _ = extract_solution(output_string)
            if check_result(output_string, output_extracted, answer) == 1:
                print("success")
                new_entry = {
                    "system_prompt": system_prompt,
                    "query": query,
                    "output": output
                }
                saved_data.append(new_entry)
                with open(save_path, 'w') as f:
                    json.dump(saved_data, f, indent=2)
        else:
            failed += 1
            if failed > 5:
                print("Too many failures, stopping.")
                break
        with open(index_path, 'w') as f:
            f.write(str(idx + 1))