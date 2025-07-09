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
import random
from copy import deepcopy
import string

random.seed(12)

def replace_param_names_new(data):
    
    letters = list(string.ascii_uppercase)+list(string.ascii_lowercase)+['_','.']+list(map(str,range(10)))
    new_data = deepcopy(data)
    tools = []
    t_name= []
    old_tools = json.loads(new_data['tools'])
    N =len(old_tools)
    for t in old_tools:
        if t['name'] not in t_name:
            tools.append(t)
            t_name.append(t['name'])
    answers = json.loads(data['answers'])
    
    i = 0
    for tool in tools:
        
        keys = list(tool['parameters'].keys())
        for param in keys:
            old_name = param
            new_name = ''.join(random.choice(letters) for i in range(random.randint(4,10)))  
            tool['parameters'][new_name] = tool['parameters'].pop(old_name)
            if len(new_data['answers']):
                for answer in answers:
                    if old_name in answer['arguments'] and answer['name']==tool["name"]:
                        answer['arguments'][new_name] = answer['arguments'].pop(old_name)
    if len(tools)!=N:
        tools=tools+random.choices(tools,k=random.randint(0,N-len(tools)+1))
    random.shuffle(tools)
    new_data['tools'] = json.dumps(tools)
    new_data['answers'] = json.dumps(answers)

    return new_data


def replace_function_names_new(data):
    letters = list(string.ascii_uppercase)+list(string.ascii_lowercase)+['_','.']+list(map(str,range(10)))
    answers = json.loads(data['answers'])
    
    new_data = deepcopy(data)
    tools = []
    t_name= []
    old_tools = json.loads(new_data['tools'])
    N =len(old_tools)
    for t in old_tools:
        if t['name'] not in t_name:
            tools.append(t)
            t_name.append(t['name'])
    i = 0
    for tool in tools:
        old_name = tool['name']
        new_name = new_name = ''.join(random.choices(letters,k=random.randint(5,15))) 
        tool['name'] = new_name
        if len(new_data['answers']):
            for answer in answers:
                if answer['name'] == old_name:
                    answer['name'] = new_name

        i+=1
    if len(tools)!=N:
        tools=tools+random.choices(tools,k=random.randint(0,N-len(tools)+1))
    random.shuffle(tools)
    new_data['tools'] = json.dumps(tools)
    new_data['answers'] = json.dumps(answers)
    
    return new_data

def generate_random_value(defalt):
    if type(defalt)!=str:
        if type(defalt) == int:
            defalt+=random.randint(1,4)
        else:
            defalt+=random.random()
        return defalt
    letters = list(string.ascii_uppercase) + list(string.ascii_lowercase) + ['_', '.'] + list(map(str, range(10)))
    return ''.join(random.choice(letters) for _ in range(5))

def replace_in_query(query, old_value, new_value):
    old_value, new_value = str(old_value), str(new_value)
    query = query.replace(old_value, new_value)
    query = query.replace(old_value.capitalize(), new_value.capitalize())
    query = query.replace(old_value.lower(), new_value.lower())
    query = query.replace(old_value.upper(), new_value.upper())
    return query

def replace_in_des(des, old_value, new_value):
    des = des.replace(str(old_value), str(new_value))

    return des

def replace_param_default_values_news(data):
    new_data = deepcopy(data)
    tools = []
    t_name= []
    old_tools = json.loads(new_data['tools'])
    N =len(old_tools)
    for t in old_tools:
        if t['name'] not in t_name:
            tools.append(t)
            t_name.append(t['name'])
    answers = json.loads(new_data['answers'])

    for tool in tools:
        
        for param, param_info in tool['parameters'].items():
            default_value = param_info.get('default', None)
            
            if type(default_value) == list:
                continue
            if default_value==None or default_value=='':
                continue
            
            keep = 0
            for ans in answers:
                if ans['name'] != tool['name']:
                    continue
                
                if param in ans["arguments"] and str(default_value)==str(ans["arguments"][param]):
                    keep=1

                    break
                
            if keep==0:
                continue
                        
            # Randomly generate a new default value
            new_default_value = generate_random_value(deepcopy(default_value))
            
            # Replace in tool's parameters
            tool['parameters'][param]['default'] = new_default_value
            tool['parameters'][param]['description'] = replace_in_des(tool['parameters'][param]['description'], default_value, new_default_value)
            # Replace in answers
            for answer in answers:
                if answer['name'] == tool['name'] and param in answer['arguments']:
                    argument_value = answer['arguments'][param]
                    if default_value==argument_value:
                        answer['arguments'][param] = new_default_value
                        
                        new_data['query'] = replace_in_query(new_data['query'], default_value, new_default_value)
    
    new_data['tools'] = json.dumps(tools)
    new_data['answers'] = json.dumps(answers)
    
    return new_data

def check_default_values_news(data):
    new_data = deepcopy(data)
    tools = []
    t_name= []
    old_tools = json.loads(new_data['tools'])
    N =len(old_tools)
    for t in old_tools:
        if t['name'] not in t_name:
            tools.append(t)
            t_name.append(t['name'])
    answers = json.loads(new_data['answers'])


    for tool in tools:
        
        for param, param_info in tool['parameters'].items():
            default_value = param_info.get('default', None)
            
            if type(default_value) == list:
                continue
            if default_value==None or default_value=='':
                continue
            
            keep = 0
            for ans in answers:
                if ans['name'] != tool['name']:
                    continue
                
                if param in ans["arguments"] and str(default_value)==str(ans["arguments"][param]) and str(default_value).lower() not in data['query'].lower():
                    keep=1

                    break
                
            if keep==0:
                continue
            else:
                return True
    return False
