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

from prompts.action_prompts import action_dict
from dataset.neuropixel import dataset_details, load_data, load_video_to_numpy

from utils.string_utils import *
from utils.history_utils import *
from utils.code_utils import *

# put data in globals to help with running llm generated code that assumes this exists 
exec("data = load_data()", globals()) # hacky

# code debugging and execution utils
def exec_and_get_error(code):
   try:
      exec(code, globals())
      return True, ""
   except Exception as e:
      tb = traceback.format_exc()
      return False, tb