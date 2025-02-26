# recursive-scientist

Reasoning models seem to be able to solve almost all small, well-defined scientific tasks remarkably well, so one approach to developing an AI Scientist using an agentic approach could be to figure out how to flexibly 1) decompose a large task into a series of small, well-defined subtasks, 2) identify and provide the necessary context & tools to an LLM to solve the subtask, and 3) chain together the results of the subtasks to fulfill the larger goal.  

To illustrate this framework, I developed a minimal pilot implementation of an agent that performs the ‘decomposition step’ recursively at all stages of the scientific process.  I have not implemented solutions for (2) and (3) and this is still a work in progress, so the functionality of the pilot is very limited. Nonetheless, examining how the agent tries to decompose large tasks into subtasks is informative.

The key component of this framework is a ‘Task’ class with a ‘select_action’ method where a model identifies the best action from a list of potential actions given its current objective, a ‘take_action’ method where the LLM receives an action specific prompt and executes the chosen action, and a ‘summarize’ method where the LLM summarizes what it has done so that the next subtask can use and build upon its results. Crucially, one of the actions the LLM can select from is ‘divide the task into smaller subtasks’ which allows it to spawn new children to solve subtasks rather than trying to tackle the entire task at once. Given a general base prompt such as, “make a new scientific discovery using this dataset”, and information about a dataset, the agent identifies subtasks and spawns children to address them. Pseudocode for this is shown below:
```
def process_subtasks(task):
    for subtask in task.subtasks:
        subtask.select_action()
        subtask.take_action()
        process_subtasks(subtask)

# instantiate root task 
task=Task(task_prompt, dataset_prompt, action_dict, model_name='deepseek-r1', level=0, max_depth=3)

# take two seed actions
task.take_action("Brainstorm")
task.take_action("Divide task into smaller subtasks")

# recurse
process_subtasks(task)
```

The keys in the action dictionary are the potential actions the model can take, e.g., query papers, write code, summarize results, think about a problem, divide a problem into subproblems etc., and the values in the action dictionary are the different prompts associated with those actions. An abbreviated action dictionary with three actions is shown below:

```
brainstorm = {
"select_action": "Brainstorm ideas about potential hypotheses and analyses to do with the dataset.", 
"take_action": "Instructions:\n Generate a JSON-formatted list of string of the different ideas/hypotheses/analyses. Do not include any idea entries that are not possible given the dataset and tools you are provided with."
}

code = {
"select_action": "Write Python code to preprocess the data, conduct an analysis, and/or make a figure", 
"take_action": "Write Python code to complete the task, following the specifications below. \nInstructions:\n code formatting details..."
}

divide = {
"select_action": "If the task is too large or challenging to address with any one of the other options, select this action to make the task simpler by breaking it into subtasks.", 
"take_action": "You have decided to begin addressing this task by breaking it down into smaller subtasks. Think deeply about what subtasks you need to do to acheive the overall goal, and identify a sequence of subtasks that should be performed to acheive the goal. The subtasks should be ordered. The first subtasks that you do should help you with completing later subtasks. Ensure that by the final subtask, you will have completed the overall goal. \nInstructions:\n json formatting details..."
}

action_dict = OrderedDict({"Brainstorm": brainstorm,
               "Write code": code, 
               "Divide task into smaller subtasks": divide})
```
