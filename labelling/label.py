from typing import NamedTuple, List, Optional, Tuple
import time
import pygame
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
FPS = 60
FWHM = 9
TRIAL_COUNT = 50
TARGET_FREQ = 0.9

# Types
class Trial(NamedTuple):
    """List entry for the `trials` list."""
    is_mountain: bool
    responses: List[float]

# Initialize pygame
pygame.init()

# Set up the window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('gradCPT')

# Load images
city_images = [pygame.image.load(f'labelling/images/city/city_{i}.jpg') for i in range(10)]
mountain_images = [pygame.image.load(f'labelling/images/mountains/mountain_{i}.jpg') for i in range(10)]


def record_responses() -> Tuple[List[Trial], float, float]:
    """Present images to the user and records raw response times."""
    clock = pygame.time.Clock()
    trials = [{'is_mountain': False, 'responses': []} for _ in range(TRIAL_COUNT)]

    # Change this value to make it 800ms on your machine. This value is for a M1 Macbook Air.
    alpha_change_per_frame = 256 / 33

    cur_img, is_mountain = get_image()

    start_timestamp = time.time()
    for i in range(TRIAL_COUNT):
        trials[i]['is_mountain'] = is_mountain
        next_img, next_is_mountain = get_image(cur_img)
        trial_start = time.time()
        for alpha in np.arange(0, 256, alpha_change_per_frame):
            cur_img.set_alpha(255 - alpha)
            next_img.set_alpha(alpha)
            screen.blit(cur_img, (0, 0))
            screen.blit(next_img, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    trials[i]['responses'].append((time.time() - trial_start) * 1000)

        cur_img, is_mountain = next_img, next_is_mountain

    end_timestamp = time.time()
    return start_timestamp, end_timestamp, trials

def process_responses(trials: List[Trial]) -> List[Optional[float]]:
    """Calculate response times (RTs) from the trial data."""
    response_times = [float('inf')] * TRIAL_COUNT

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

def label(response_times: List[Optional[float]]) -> List[int]:
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
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    vtc_smoothed = gaussian_filter1d(vtc, sigma=sigma)  # sigma derived from FWHM

    # Determine "in the zone" (1) and "out of the zone" (0) labels
    median_vtc = np.median(vtc_smoothed)
    zone_labels = [1 if value <= median_vtc else 0 for value in vtc_smoothed]

    return zone_labels

def get_image(last_image: pygame.Surface = None) -> Tuple[pygame.Surface, bool]:
    """Returns an image depending on the last image shown."""
    is_mountain = False
    # Initial decision whether to pick from city_images or mountain_images
    if np.random.rand() < TARGET_FREQ or (last_image and last_image in mountain_images):
        pool = city_images
    else:
        pool = mountain_images
        is_mountain = True

    # Now select an image, but ensure it's not the same as last_image
    choice = np.random.choice(pool)
    while choice == last_image:
        choice = np.random.choice(pool)

    return choice, is_mountain

if __name__ == "__main__":
    _, _, raw_responses = record_responses()
    responses = process_responses(raw_responses)
    labels = label(responses)
    print(labels)
