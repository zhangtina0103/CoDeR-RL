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


import json

def truncate_after_first_tool_call_end(s):
    marker = "</tool_call>"
    index = s.find(marker)
    if index != -1:
        return s[:index + len(marker)]
    return s

def truncate_before_expert_prompt(s):
    marker = "You are an expert in composing functions"
    index = s.find(marker)
    if index != -1:
        return s[index:]
    return s

# Define file paths
input_file = "path/to/distilled_data/sft_data.json"
output_file = "path/to/data/tool_sft.json"

# Load the input JSON data
with open(input_file, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# Transform the data to match the Alpaca format
alpaca_data = []
for item in input_data:
    transformed_item = {
        "system_prompt": truncate_before_expert_prompt(item.get("system_prompt", "")),
        "query": item.get("query", ""),
        "output": truncate_after_first_tool_call_end(item.get("output", "")),
        "instruction": ""
    }
    alpaca_data.append(transformed_item)

# Save the transformed data to the output file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

print(f"Data has been successfully transformed and saved to {output_file}")