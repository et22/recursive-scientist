import re
import json
import ast
from typing import List, Dict
import traceback
import subprocess
import sys
import logging
import os
import pickle
import datetime

from model.llm import TextLLM, VisionLLM
from collections import OrderedDict
from copy import deepcopy

from utils.string_utils import *
from utils.history_utils import *
from utils.code_utils import *

class Task:
    """
    Class representing a task in the scientific research process.
    """
    def __init__(self, level, task_prompt, action_dict, comment_list, idea_list, dataset_info, logging, model_name='deepseek-r1:32b', max_depth=2):
        self.level = level
        self.task_prompt = task_prompt
        self.action_dict = action_dict
        self.action_list = list(action_dict.keys())
        self.action = None

        self.comment_list = comment_list 
        self.idea_list = idea_list 
        self.dataset_info = dataset_info

        self.model_name = model_name
        self.lm = TextLLM(model_name=model_name, logging=logging)

        self.n_select_retries = 5
        self.n_take_retries = 10
        self.n_debug_steps = 10
        self.n_reflections = 0
        self.max_depth = max_depth

        if self.level >= self.max_depth:
            action_dict = deepcopy(self.action_dict)
            key_to_remove =  "Divide task into smaller subtasks"
            if key_to_remove in action_dict.keys():
                del action_dict[key_to_remove]

        self.failed = False
        self.result = None
        self.subtasks = []
        
        self.logging = logging
        
    def select_action(self, retry_cnt=0):
        self.logging.info("selecting action for a node at " + str(self.level))
        
        self.history_prompt = get_current_history(self.idea_list, self.dataset_info, self.comment_list)
        
        action_start_prompt = "You are thinking about the following task. " 
        action_end_prompt =  ('''Instructions: Your current goal is to identify the best action to take to make progress on solving the task (but do not do the action). 
                    Think about which of the following choices is best to solve the task:\n''' +\
                    "\n".join([str(i+1) + ". " + self.action_list[i] + ": " + self.action_dict[self.action_list[i]]['select_action'] for i in range(len(self.action_list))]) +\
                    "\nInstruction: Your output should only contain the integer corresponding to the action you select enclosed in ** **, and a one sentence justification of your choice. Do not do the action that you selected. Do not write Python code.")
                               
        prompt = combine_prompts([action_start_prompt, self.task_prompt, self.history_prompt, action_end_prompt])

        action_select_response = self.lm.chat_with_model(prompt)

        chosen_action = parse_action_selection(action_select_response)
        
        if (chosen_action == -1 or chosen_action > len(self.action_list) or chosen_action == 0):
            #if parsing failed, call the llm again
            if retry_cnt < self.n_select_retries:
                self.select_action(retry_cnt=retry_cnt+1)
            else:
                self.action = self.action_list[0] # brainstorm if we're struggling with parsing for some reason
                self.logging.info("LM action failed to select, using: " + self.action)
        else:
            # if parsing worked set action
            self.action = self.action_list[chosen_action - 1] # we added one for the list numbering beacuse model probably more used to indexing from 1, so subtracing one here 
            self.logging.info("LM action selected: " + self.action)

    def take_action(self, retry_cnt=0):
        self.logging.info("taking action " + self.action + " for a node at " + str(self.level) + "...")
        
        if "brainstorm" in self.action.lower():
            self.history_prompt = get_current_history(self.idea_list, self.dataset_info, self.comment_list, include_code=False)
        elif "code" in self.action.lower():
            self.history_prompt = get_current_history(self.idea_list, self.dataset_info, self.comment_list, include_results=False, include_ideas=False)
        elif "divide" in self.action.lower():
            self.history_prompt = get_current_history(self.idea_list, self.dataset_info, self.comment_list, include_ideas=True, include_code=False)
        
        action_start_prompt = self.action_dict[self.action]['take_action_start']
        action_end_prompt = self.action_dict[self.action]['take_action_end']
        
        prompt = combine_prompts([action_start_prompt + self.task_prompt, self.history_prompt, action_end_prompt])
        action_implement_response = self.lm.chat_with_model(prompt)
        
        for i in range(self.n_reflections):
            prompt = self.action_dict[self.action]['take_action_revise']
            action_implement_response = self.lm.chat_with_model(prompt)

        try:
            parsed_response = self.parse_action_response(action_implement_response)
        except Exception as e:
            logging.info(f"error when parsing action response: {e}")
            parsed_response = None
        
        if parsed_response is None:
            # if parsing failed, call the llm again
            if retry_cnt < self.n_take_retries:
                logging.info("parsing failed... retrying")
                self.take_action(retry_cnt=retry_cnt+1)
                return
            else:
                # TODO - maybe need to do other things here
                logging.info(f"Note - take action is struggling with parsing the following model output: {action_implement_response}")
                return

        self.implement_action(parsed_response)

    def parse_action_response(self, response):
        if "brainstorm" in self.action.lower():
            return parse_action_brainstorm(response)
        elif "code" in self.action.lower():
            return parse_action_code(response)
        elif "divide" in self.action.lower():
            return parse_action_divide(response)
            
    def implement_action(self, parsed_response):
        if "brainstorm" in self.action.lower():
            self.idea_list += parsed_response
        elif "code" in self.action.lower():
            self.logging.info("executing and debugging code...")
            code = self.exec_debug_code(parsed_response)
            if not self.failed:
                self.logging.info("code ran succesfully! parsing the code...")

                # log all our python functions in one file
                curr_function_dict = parse_functions(code)
                for key in curr_function_dict:
                    with open("./logs/code_log.py", "a") as file:
                        file.write("\n\n\n")
                        file.write(curr_function_dict[key])
                                    
                curr_comment_list = parse_comments(code)
                self.comment_list += curr_comment_list
        elif "divide" in self.action.lower():
            self.logging.info("splitting tasks...")
            self.subtasks = []
            new_task = None
            task_prompts = []
            for task_def in parsed_response:
                if 'summary' in task_def.keys() and 'description' in task_def.keys():
                    task_prompt = ". ".join([task_def['summary'], task_def['description']])    
                else:
                    self.logging.info(f'summary and description not in json {task_def}')
                    task_prompt = ""
                    for key in task_def.keys():
                        task_prompt += "\n" + task_def[key] + "\n"
                task_prompts.append(task_prompt)
                
            idea_list = task_prompts    # idea list shared among the next level of nodes
            for task_prompt in task_prompts:
                if self.level + 1 == self.max_depth:
                    action_dict = deepcopy(self.action_dict)
                    key_to_remove =  "Divide task into smaller subtasks"
                    if key_to_remove in action_dict.keys():
                        del action_dict[key_to_remove]
                else:
                    action_dict = self.action_dict
                new_task = Task(self.level+1, task_prompt, action_dict, self.comment_list, idea_list, self.dataset_info, self.logging,  model_name=self.model_name, max_depth=self.max_depth)
                self.subtasks.append(new_task)
        self.summarize_result()

    def exec_debug_code(self, code):
        debug_cnt = 0
        status = False
        default_error_prompt = self.action_dict[self.action]['debug_default'] # this is a function
        mnf_error_prompt = self.action_dict[self.action]['debug_mnf'] # this is a function
        while debug_cnt < self.n_debug_steps and not status:
            # print("executing code...")
            status, output = exec_and_get_error(code)
            if not status:
                # print("fixing error...")
                if 'ModuleNotFoundError' in output:
                    error_response = self.lm.chat_with_model(mnf_error_prompt(output))
                    exec(parse_action_code(error_response), globals())
                else:
                    code = parse_action_code(self.lm.chat_with_model(default_error_prompt(output, code)))
            debug_cnt += 1
        
        if not status:
            self.failed = True
        return code

    def summarize_result(self):
        summarize_prompt = ("Your task was to: " + self.task_prompt + "\n\n"
                            "Please summarize what you have done to acheive that goal, and include all details that might be relevant "
                             "for your colleague to build upon what you have worked on. For example, if you wrote code, describe how it should be used. "  
                             "If you saved any output plots, describe where they are saved and what they contain. "
                             "If you saved any statistical results text, describe where they are saved and what they contain.")

        if self.result is None:
            response = self.lm.chat_with_model(summarize_prompt)
            self.result = response
        