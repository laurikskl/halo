import time
import pygame
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
FPS = 60
TRANSITION_TIME = 800  # in ms
GAUSSIAN_KERNEL_WIDTH = 7  # in seconds
FWHM = 9
TRIAL_COUNT = 10
TARGET_FREQ = 0.9

# Initialize pygame
pygame.init()

# Set up the window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('gradCPT')

# Load images
city_images = [pygame.image.load(f'labelling/images/city/city_{i}.jpg') for i in range(10)]
mountain_images = [pygame.image.load(f'labelling/images/mountains/mountain_{i}.jpg') for i in range(10)]


def gather_responses():
    """Runs the gradCPT task."""
    clock = pygame.time.Clock()
    trials = [{'is_mountain': False, 'responses': []} for _ in range(TRIAL_COUNT)]
    start_timestamp = time.time()

    # Change this value to make it 800ms on your machine. This value is for a M1 Macbook Air.
    alpha_change_per_frame = 256 / 33

    cur_img, is_mountain = get_image()

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
        trial_end = time.time()
        print(f"Time taken: {trial_end - trial_start}")

    end_timestamp = time.time()
    return start_timestamp, end_timestamp, trials

def process_responses(trials):
    """Calculate response times from the trial data."""
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
    
    return response_times

def calculate_rtv(response_times):
    """Calculate RTV aka the trial to trial variation in response time"""
    # Z-normalized: normalized such that mean is 0 and std is 1
    z_normalized_rt = (response_times - np.mean(response_times)) / np.std(response_times)
    vtc = np.abs(z_normalized_rt - np.mean(response_times))

    # Linear interpolation for missing values
    # Filling gaps in a simple way, assuming gaps are only one trial wide
    for i in range(1, len(vtc) - 1):
        if np.isnan(vtc[i]):
            vtc[i] = (vtc[i-1] + vtc[i+1]) / 2

    # Smooth the VTC
    vtc_smoothed = gaussian_filter1d(vtc, sigma=7)  # sigma derived from FWHM

    return vtc_smoothed

def get_image(last_image: pygame.Surface = None):
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

def test():
    test_responses = [{'is_mountain': False, 'responses': [400]}, {'is_mountain': False, 'responses': [100, 400, 780]}, {'is_mountain': False, 'responses': [200, 570]}, {'is_mountain': False, 'responses': [400]}, {'is_mountain': False, 'responses': []}, {'is_mountain': False, 'responses': [20, 780]}, {'is_mountain': False, 'responses': []}, {'is_mountain': False, 'responses': [400]}]
    test_rts = process_responses(test_responses)
    print(test_rts)


if __name__ == "__main__":
    test()
