import logging
import os
import pickle
import datetime

from prompts.action_prompts import action_dict
from dataset.neuropixel import dataset_details, load_data, load_video_to_numpy
from model.task import Task

def setup_logging():
    log_file = './logs/recurse_log.txt'

    if os.path.exists(log_file):
        os.remove(log_file)
        
    # Ensure the directory exists (if logging to a subdirectory)
    os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
    
    # Configure logging to write to a file
    logging.basicConfig(
        filename=log_file,  # Log file name
        level=logging.INFO,  # Log level (INFO, DEBUG, WARNING, etc.)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        filemode='w'  # Overwrite the file each time (use 'a' to append)
    )
    return logging

def process_subtasks(task):
    for subtask in task.subtasks:
        subtask.select_action()
        subtask.take_action()
        process_subtasks(subtask)

def main():    
    base_prompt = "Your overall goal is to make novel scientific discoveries about a dataset you are provided with."
    logging = setup_logging()

    task=Task(level=0, task_prompt=base_prompt, action_dict=action_dict, comment_list=[], idea_list=[], dataset_info=dataset_details, logging=logging, model_name='deepseek-r1', max_depth=2)

    # take two seed actions
    task.action = "Brainstorm"
    task.take_action()
    
    task.action = "Divide task into smaller subtasks"
    task.take_action()

    # recurse
    process_subtasks(task)
    
    # Save the task object with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"./logs/task_state_{timestamp}.pkl"
    print(f"\nSaving task state to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(task, f)
    print("Task state saved successfully!")

if __name__ == "__main__":
    main()
    




