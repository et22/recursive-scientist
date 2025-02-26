import re
import json
import ast
from typing import List, Dict
import traceback
import subprocess
import sys

from model.llm import TextLLM, VisionLLM
from collections import OrderedDict
from copy import deepcopy

from collections import OrderedDict

# code specific prompts + functions
code_format = ("The Python code should be enclosed within triple backticks ``` ```. " 
                   "You MUST make at least one plot in the code, and save it with plt.savefig to the ./outputs/ directory." 
                    "You MUST save at least one numeric/statistical result related to the task in a text file with file.write(...) to the ./outputs/ directory." 
                    "ALWAYS use the REAL data not simulated data. The variable 'data' is provided to you as a global variable which has already been loaded."
                   "Please write code in multiple functions when possible. Any function should include a comment at the beginning containg a description of the function, and the expected input(s) and output(s), including their types.")

"""
default_error_prompt = lambda output, code: f"Your task is to debug Python code by correcting the error while changing as little in the code as possible. The error with the code is: {output}.\n The original python code is: {code}. \n You should return the complete revised code with the error fixed.\n" + \
    code_format
"""
default_error_prompt = lambda output, code: f"Your code returned this error {output}. Please fix it and return the complete code with the error fixed."

mnf_error_prompt = lambda output: f"My Python code returned the following error: {output}. Please return Python code enclosed in in ``` ``` tags that can install the appropriate module to fix this issue with pip. Use this template for running pip from Python code:\n" + \
'''
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install","--no-cache-dir", package])
'''


json_example = '''[
    {
        "summary": "Write a project proposal",
        "description": "Draft a comprehensive project proposal outlining the objectives, scope, timeline, and expected outcomes. Include background research, methodology, and resource requirements to ensure clarity for stakeholders."
    },
    {
        "summary": "Collect and preprocess data",
        "description": "Gather relevant datasets from multiple sources and clean the data by handling missing values, removing duplicates, and normalizing formats. Perform exploratory data analysis to identify trends and potential issues."
    },
    {
        "summary": "Write test cases",
        "description": "Design and implement unit and integration tests to ensure code reliability and functionality. Automate testing where possible and document edge cases to improve overall software robustness."
    },
]'''

# action prompts
brainstorm_dict = {"select_action": "Brainstorm ideas about potential hypotheses and analyses to do with the dataset.", 
                   "take_action_start": "Your task is to brainstorm ideas related to the following task. Ensure that each idea you come up with is testable: ", 
                   "take_action_end": ("Instructions:\n"
                                      "Generate a JSON-formatted list of strings of the different ideas/hypotheses/analyses. Each string in the list should be at least 2 sentences long. Do not include any idea entries that are not possible given the dataset and tools you are provided with. The tools you have are the ability to write and implement code, look at plots and read statistical results, and summarize your findings. Ensure your response is complete and comprehensive. The output should be strictly in JSON format, containing only the list of strings without any additional text."),
                   "take_action_revise": (
                                "Please reflect on and revise your previous response to ensure it is complete and comprehensive. " + \
                                "Delete any idea entries that are not possible given the dataset and tools you are provided with. " +\
                                "The tools you have are the ability to write and implement code, look at plots and read statistical results, and summarize your findings." +\
                                "The output should be strictly in JSON format, containing only the JSON data without any additional text.")}

code_dict = {"select_action": "Write Python code to preprocess the data, conduct an analysis, and/or make a figure", 
               "take_action_start": "Your task is to write Python code to accomplish the following task: ", 
               "take_action_end": ("You should write Python code to complete the task, following the specifications below. \nInstructions:\n" + code_format), 
                "take_action_revise": (
                    "Please reflect on and revise your previous response to ensure it is complete and comprehensive."
                    "Ensure that the output meets the correct specification, provided below: \n\n") + code_format, 
                "debug_default": default_error_prompt, 
                 "debug_mnf": mnf_error_prompt, 
            }

divide_dict = {"select_action": "If the task is too large or challenging to address with any one of the other options, select this action to make the task simpler by breaking it into subtasks.",
                "take_action_start": "Your task is to break down the following task into smaller subtasks: ", 
                "take_action_end": (
                    "You have decided to begin addressing this task by breaking it down into smaller subtasks. Think deeply about what subtasks you need to do to acheive the overall goal, and identify a sequence of subtasks that should be performed to acheive the goal."  
                    "The subtasks should be ordered. The first subtasks that you do should help you with completing later subtasks. Ensure that by the final subtask, you will have completed the overall goal. \n\n"
                    "Instructions:\n"
                    "Generate a JSON-formatted list of tasks you would need to perform. Each task should be represented as a dictionary with two keys:\n"
                    "summary: A one-sentence summary of the task.\n"
                    "description: A thorough, multi-sentence explanation of the task.\n"
                    "The output should be strictly in JSON format, containing only the JSON data without any additional text.\n"
                    "An example of the output format is: \n"
                    ) + json_example, 
               "take_action_revise": (
                    "Please reflect on and revise your previous response to ensure it is complete and comprehensive. " + \
                    "Delete any subtask entries that are not possible given the dataset and tools you are provided with. " +\
                    "Add subtasks that could help link the output of one subtask with the input of the next." +\
                    "The tools you have are the ability to write and implement code, look at plots and read statistical results, and summarize your findings." +\
                    "The output should be strictly in JSON format, containing only the JSON data without any additional text."
                    "An example of the output format is: \n"
                    )+json_example}

action_dict = OrderedDict({"Brainstorm": brainstorm_dict,
               "Write code": code_dict, 
               "Divide task into smaller subtasks": divide_dict})