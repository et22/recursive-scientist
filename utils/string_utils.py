import re
import json
import ast
from typing import List, Dict
import traceback
import subprocess
import sys

from utils.string_utils import *
from utils.history_utils import *
from utils.code_utils import *

import textwrap
import re

def remove_main_guard(code: str) -> str:
    pattern = r'if __name__ == "__main__":\n(.*)'  # Match the main guard
    match = re.search(pattern, code, re.DOTALL)
    
    if match:
        indented_block = match.group(1)
        unindented_block = textwrap.dedent(indented_block)
        code = re.sub(pattern, unindented_block, code, flags=re.DOTALL)
    
    return code.strip()
    
def parse_comments(code: str) -> List[Dict[str, str]]:
    """
    Parse a string containing Python code and extract function names, input variable comments,
    and output comments.

    Args:
        code (str): A string containing Python code.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary contains:
            - "name": The name of the function.
            - "inputs": A string describing the input variables (extracted from comments).
            - "output": A string describing the output (extracted from comments).
    """
    # Parse the code into an AST
    tree = ast.parse(code)

    # Initialize a list to store function metadata
    functions = []

    # Iterate through the AST nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Extract the function name
            function_name = node.name

            # Extract the function's docstring (if any)
            docstring = ast.get_docstring(node)

            # Initialize input and output descriptions
            description = ''
            inputs = ''
            outputs = ''

            # Parse the docstring for input and output descriptions
            if docstring:
                lines = docstring.split("\n")
                phase = 'description'

                for line in lines:
                    if line.strip().lower().startswith("input"):
                        phase = 'input'
                    elif line.strip().lower().startswith("arg"):
                        phase = 'input'
                    elif line.strip().lower().startswith("parameter"):
                        phase = 'input'
                    elif line.strip().lower().startswith("output"):
                        phase = 'output'
                    elif line.strip().lower().startswith("return"):
                        phase = 'output'
                        
                    if phase == 'description':
                        description += line + "\n"
                    elif phase == 'input':
                        inputs += line+ "\n"
                    elif phase == 'output':
                        outputs += line+ "\n"

            # Add the function metadata to the list
            functions.append({
                "name": function_name,
                "description": description,
                "input": inputs,
                "output": outputs,
                "docstring": docstring
            })
            
    return functions

def parse_functions(code):
    tree = ast.parse(code)
    functions = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            func_start = node.lineno - 1
            func_end = max(n.lineno for n in ast.walk(node) if hasattr(n, 'lineno'))
            
            func_lines = code.split('\n')[func_start:func_end]
            functions[func_name] = '\n'.join(func_lines)
    
    return functions
    
# string processing utils
def parse_action_selection(s):
    """
    Extracts an integer enclosed in double asterisks (**) from a string, even if there is text in between.
    
    Input: s (str) - The input string potentially containing **integer** patterns.
    Output: int - The extracted integer or -1 if no valid number is found.
    """
    match = re.search(r'\*\*[^\d]*(\d+)[^\d]*\*\*', s)
    if match:
        return int(match.group(1))
    return -1  
    
def parse_action_code(input_string):
  section = re.search(r"```(.*?)```", input_string, re.DOTALL).group(1) if re.search(r"```(.*?)```", input_string, re.DOTALL) else None
  output_string = '\n'.join(lines[1:]) if (lines := section.splitlines())[0].find('import') == -1 else section
  output_string =  remove_main_guard(output_string)
  return output_string

def parse_json(text):
    """
    Extracts and parses the JSON portion from a given text containing both text and JSON.
    
    Input: text (str) - The full text output from the LLM, including embedded JSON.
    Output: list[dict] - A list of dictionaries extracted from the 'subtasks' field in the JSON.
    """
    try:
        # Extract the JSON portion enclosed in triple backticks
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if not match:
            #print("Error: No JSON found in text.")
            return None
        
        json_str = match.group(1)  # Extract JSON content
        json_data = json.loads(json_str)  # Parse JSON
        
        # Extract 'subtasks' list if present
        return json_data
    except json.JSONDecodeError:
        #print("Error: Failed to parse JSON.")
        return None

def parse_action_divide(text):
    return parse_json(text)

def parse_action_brainstorm(text):
    return parse_json(text)

def combine_prompts(prompt_list):
    return "\n\n".join(prompt_list)
