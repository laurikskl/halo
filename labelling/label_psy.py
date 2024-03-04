from typing import TypedDict, Optional, Tuple
from time import perf_counter
from psychopy import visual, core, event
from scipy.ndimage import gaussian_filter1d
import numpy as np
import random
import os

# constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
IMAGE_DISPLAY_TIME = 0.8  # 800 ms
TARGET_FREQ = 0.9
TRIAL_N = 50 # 75 trials = 1 minute
FWHM = 9
SIGMA = FWHM / (2 * np.sqrt(2 * np.log(2)))

# Types
class Trial(TypedDict):
    """List entry for the `trials` list."""
    is_mountain: bool
    responses: list[float] 

# Paths to images
city_images = [f'images/city/city_{i}.jpg' for i in range(10)]
mountain_images = [f'images/mountains/mountain_{i}.jpg' for i in range(10)]

# Setup the window
win = visual.Window(size=(SCREEN_WIDTH, SCREEN_HEIGHT), fullscr=False)

# Preload all images
preloaded_city_images = [visual.ImageStim(win, image=img, size=(1, 1)) for img in city_images]
preloaded_mountain_images = [visual.ImageStim(win, image=img, size=(1, 1)) for img in mountain_images]

def transitionImages(from_img: visual.ImageStim, to_img:visual.ImageStim) -> list[float]:
    transition_clock = core.Clock()
    DISPLAY_TIME_WITH_BUFFER = IMAGE_DISPLAY_TIME - 0.05
    
    while transition_clock.getTime() < DISPLAY_TIME_WITH_BUFFER:
        weight = transition_clock.getTime() / DISPLAY_TIME_WITH_BUFFER
        from_img.opacity = 1 - weight
        to_img.opacity = weight
        from_img.draw()
        to_img.draw()
        win.flip()
    
    elapsed_time = transition_clock.getTime()
    remaining_time = IMAGE_DISPLAY_TIME - elapsed_time
    if remaining_time > 0:
        core.wait(remaining_time)
    
    presses = event.getKeys(keyList=['space'], timeStamped=transition_clock)
    timestamps_ms = [timestamp * 1000 for (_, timestamp) in presses]
    
    return timestamps_ms

    

def get_image(last_image:visual.ImageStim = None) -> Tuple[visual.ImageStim, bool]:
    """Returns an image depending on the last image shown."""
    is_mountain = False
    # Initial decision whether to pick from city_images or mountain_images
    if random.random() < TARGET_FREQ or (last_image and last_image in preloaded_mountain_images):
        pool = preloaded_city_images
    else:
        pool = preloaded_mountain_images
        is_mountain = True

    # Now select an image, but ensure it's not the same as last_image
    choice = random.choice(pool)
    while choice == last_image:
        choice = random.choice(pool)

    return choice, is_mountain

def record_responses() -> Tuple[float, float, list[Trial]]:
    trials = []
    
    last_image, _ = get_image()
    start_timestamp = perf_counter()
    for i in range(TRIAL_N):
        trial_clock = core.Clock()
        next_image, is_mountain = get_image(last_image)
        responses = transitionImages(last_image, next_image)
        trials.append({'is_mountain': is_mountain, 'responses': responses})
        last_image = next_image
        
        print(f"Trial duration: {trial_clock.getTime()} s")
    
    end_timestamp = perf_counter()
    return start_timestamp, end_timestamp, trials

def process_responses(trials: list[Trial]) -> list[Optional[float]]:
    """Calculate response times (RTs) from the trial data."""
    response_times = [float('inf')] * TRIAL_N

    # Edge case: first trial
    if trials[0]['responses']:
        response_times[0] = trials[0]['responses'][0]

    # Loop 0: unamibiguous correct responses
    for i, trial in enumerate(trials[1:], start=1):
        remaining_responses = []
        for rt in trial['responses']:
            if rt < 320 and not trials[i-1]['is_mountain']:
                response_times[i-1] = min(800 + rt, response_times[i-1])
            elif rt > 560 and not trial['is_mountain']:
                response_times[i] = min(rt, response_times[i])
            else:
                remaining_responses.append(rt)
        trial['responses'] = remaining_responses


    # Loop 1: ambigous presses
    for i, trial in enumerate(trials[1:], start=1):
        for rt in trial['responses']:
            if response_times[i-1] == float('inf') and response_times[i] != float('inf'):
                response_times[i-1] = 800 + rt
            elif response_times[i-1] != float('inf') and response_times[i] == float('inf'):
                response_times[i] = rt
            elif response_times[i-1] == float('inf') and response_times[i] == float('inf'):
                if trials[i-1]['is_mountain']:
                    response_times[i] = rt
                elif trial['is_mountain']:
                    response_times[i-1] = 800 + rt
                else:
                    if rt < 400:
                        response_times[i-1] = 800 + rt
                    else:
                        response_times[i] = rt

    # Replace inf with None
    return [None if x == float('inf') else x for x in response_times]

def label(response_times: list[Optional[float]]) -> list[int]:
    """Label responses w.r.t RTV aka the trial to trial variation in response time"""
    response_times = np.array(response_times, dtype=float)

    # Z-tranform the sequence
    z_normalized_rt = (response_times - np.nanmean(response_times)) / np.nanstd(response_times)

    # Calculate variance time course
    vtc = np.abs(z_normalized_rt - np.nanmean(z_normalized_rt))

    # Linearly interpolate missing values in the vtc
    nans, x = np.isnan(vtc), lambda z: z.nonzero()[0]
    vtc[nans] = np.interp(x(nans), x(~nans), vtc[~nans])

    # Smooth the VTC
    vtc_smoothed = gaussian_filter1d(vtc, sigma=SIGMA)  # sigma derived from FWHM

    # Determine "in the zone" (1) and "out of the zone" (0) labels
    median_vtc = np.median(vtc_smoothed)
    zone_labels = [1 if value <= median_vtc else 0 for value in vtc_smoothed]

    return zone_labels

def save_results(start_timestamp: float, end_timestamp: float, labels: list[int]):
    # Convert the list to a string with each element on a new line
    labels_str = '\n'.join(map(str, labels))

    # Convert timestamps to strings
    start_timestamp_str = str(start_timestamp)
    end_timestamp_str = str(end_timestamp)

    # Define the filename
    filename = "gradcpt_output.txt"

    # Open the file in write mode and write the data
    with open(filename, 'w') as file:
        file.write("Labels:\n")
        file.write(labels_str + "\n")
        file.write("Start Timestamp: " + start_timestamp_str + "\n")
        file.write("End Timestamp: " + end_timestamp_str + "\n")

if __name__ == "__main__":
    start_timestamp, end_timestamp, raw_responses = record_responses()
    responses = process_responses(raw_responses)
    labels = label(responses)
    save_results(start_timestamp, end_timestamp, labels)
    
# Close the window
win.close()
core.quit()

label_psy.py
