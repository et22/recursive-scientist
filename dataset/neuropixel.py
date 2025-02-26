import cv2
import numpy as np
import pickle

dataset_details = '''The dataset is from a Neuropixel recording in the middle temporal area (MT) collected while a monkey viewed 200 ms full screen natural videos. The middle temporal area is known for its role in motion processing, and for beautiful functional oranization of neurons into motion direction columns.

The data has been preprocessed for you into the following format:
1. data is a python dictionary with following keys and values - it is provided as a global variable you will always have access to. 
    key: videos, shape: (4670 trials,), dtype: object --- specifies the id of the video shown in that trial.
        e.g., array(['bc24590b-13f5-4f18-a955-23775c525df2', 
               'ca6e9ec9-46ac-4a02-8034-7c31157dc52c',...])
    key: rates, shape: (4670 trials, 965 neurons), dtype: float64 --- the firing rate for each neuron in each trial (units of spikes/sec)
    key: expvar, shape: (965 neurons,), dtype: float64 --- the 'explainable variance' of each neuron in the task, an approximate upper bound on how much variance in a neuron's response can be explained by the stimulus.
    key: split, shape: (4670 trials,), dtype: bool --- split is True if the trial is a train trial and False if it is a test trial
    key: position, shape: (965 neurons,), dtype: float32 --- the y coordinate of the neuron relative to the tip of the probe (in units micron). Smaller values for position are deeper and larger values for position are more superficial.
2. ./dataset/stimuli/ is a directory that contains the videos shown in the task. The videos are *.mp4 files. 

In each trial, one video is shown that is either a 'train' or 'test' video. Each train video is shown only once in the entire session. Each test video is shown multiple times throughout the session. Each video is a 5 frame color video of shape 360 (height) x 640 (width). 
Some of the 'neurons' may not be real neurons and should be excluded from all analyses. This can be done with criteria such as, the neuron's average firing rate should exceed 1 Hz and its explainable variance should exceed .4 or .5.

print(data.keys()) # outputs dict_keys(['videos', 'rates', 'expvar', 'split', 'position'])

If you want to load a video, this is how you could do it:  
video_path = "./dataset/stimuli/" + data['videos'][0] + ".mp4"
video = load_video_to_numpy(video_path)
print(video.shape) # outputs (5, 360, 640, 3)

'''

def load_data():
    with open('./dataset/data.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_video_to_numpy(video_path, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if there are no more frames

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)
        frame_count += 1

    cap.release()
    return np.array(frames)  # Shape: (num_frames, height, width, channels)
