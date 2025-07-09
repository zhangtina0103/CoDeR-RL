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

from bfcl.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override
import json
import re

class QwenReasonHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

    @staticmethod
    def formate_inference_function(func_list):
        result = []
        for func in func_list:
            tem = {}
            tem["type"] = "function"
            tem["function"] = func
            result.append(tem)
        str_result = ""
        for tem in result:
            str_result += json.dumps(tem)
            str_result += "\n"
        return str_result
    
    @staticmethod
    def xlam_json_to_python_tool_calls(tool_calls):
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        python_format = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                name = tool_call.get("name", "")
                arguments = tool_call.get("arguments", {})
                args_str = ", ".join(
                    [f"{key}={repr(value)}" for key, value in arguments.items()]
                )
                python_format.append(f"{name}({args_str})")

        return python_format
    
    @override
    def _format_prompt(self, messages, function):

        function_str = QwenReasonHandler.formate_inference_function(function)

        system_prompt = f"""
        You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections.

    # Tool
    Here is a list of functions in JSON format that you can invoke:

    <tools>
    {function_str}
    </tools>

    In each action step, you MUST: 
    1. think about the reasoning process in the mind before and enclosed your reasoning within <think> </think> XML tags.
    2. then return a json object with function names and arguments within <tool_call></tool_call> XML tags. i.e., <tool_call>[{{"name": <function-name>, "arguments": <args-json-object>}}, {{"name": <function-name2>, "arguments": <args-json-object2>}}, ...]</tool_call>
    3. remember complete 1 and 2 in one single reply.
    A complete reply example is: <think>To address the query, I need to send the email to Bob and then buy the banana through walmart. </think> <tool_call> [{{"name": "email", "arguments": {{"receiver": "Bob", "content": "I will bug banana through walmart"}}}}, {{"name": "walmart", "arguments": {{"input": "banana"}}}}]</tool_call>. Please make sure the type of the arguments is correct.  If no functions could be used in the current task, please make tool_calls an empty list <tool_call>[]</tool_call>"
    """
        if messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            raise ValueError("The first message should be system message")
        
        formatted_prompt = ""
        for index, message in enumerate(messages):
            if not (index != 1 and message["role"] == "assistant" and message["content"] == None):
                formatted_prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    @override
    def decode_ast(self, result, language="Python"):
        decoded_output = []
        for invoked_function in result:
            name = invoked_function["name"]
            params = invoked_function["arguments"]
            decoded_output.append({name: params})
        return decoded_output

    @override
    def decode_execute(self, result):
        if isinstance(result, list):
            tool_calls = result
        elif isinstance(result, dict):
            tool_calls = result.get("tool_calls", [])
        else:
            tool_calls = []
        function_call = self.xlam_json_to_python_tool_calls(tool_calls)
        return function_call
    
    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        def extract_solution(tool_call_str):
            pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(pattern, tool_call_str, flags=re.DOTALL)
            if not match:
                return None
            first_match = match.group(1).strip()
            try:
                return json.loads(first_match)
            except json.JSONDecodeError:
                return None
        def extract_thought(tool_call_str):
            pattern = r'<think>(.*?)</think>'
            match = re.search(pattern, tool_call_str, flags=re.DOTALL)
            if not match:
                return None
            first_match = match.group(1).strip()
            return first_match
        thought = extract_thought(api_response.choices[0].text)
        model_responses = extract_solution(api_response.choices[0].text)
        return {
            "model_responses": model_responses,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
            "thought": thought
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        def is_empty_tool(tools):
            if tools is None or len(tools) == 0:
                return True
            return False
        
        thought = model_response_data["thought"]
        tool_calls = model_response_data["model_responses"]
        if thought is not None and not is_empty_tool(tool_calls):
            content = "<think>" + thought + "</think>" + "<tool_call>" + json.dumps(tool_calls) + "</tool_call>"
        elif thought is None and not is_empty_tool(tool_calls):
            content = "<think>" + "None" + "</think>" + "<tool_call>" + json.dumps(tool_calls) + "</tool_call>"
        else:
            content = None
        inference_data["message"].append(
            {"role": "assistant", "content": content}
        )
        return inference_data
