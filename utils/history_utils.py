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

# put all of our code, idea, dataset, results history information together into a prompt 
def get_current_history(idea_list, dataset_info, comment_list, include_ideas=True, include_dataset=True, include_code=True, include_results=True):
    prompt = ""

    if include_ideas:
        try:
            if len(idea_list) > 0:
                prompt += "You have already brainstormed the following ideas:\n"
                prompt += "\n".join(idea_list)
        except: 
            pass
    if include_results:
        try: 
            with open("./results.txt", 'r', encoding='utf-8') as file:
                file_str = file.read()
    
            if len(file_str) > 10: 
                prompt += "\nYou have already identified the following results with the dataset: " + file_str
        except:
            pass

    if include_dataset:
        prompt += "\nDataset Information: " + dataset_info
        
    if include_code:
        try:
            if len(comment_list) > 0:
                prompt = "\nYou have already written the following functions, feel free to use them:\n"
                for comment in comment_list:
                    try:
                        prompt += comment['name'] + "\n" + comment['docstring'] + "\n\n"
                    except:
                        pass
        except: 
            pass

    return prompt