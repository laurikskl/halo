import pygame
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
FPS = 60
TRANSITION_TIME = 800  # in ms
GAUSSIAN_KERNEL_WIDTH = 7  # in seconds
FWHM = 9
GAME_CYCLES = 10

# Initialize pygame
pygame.init()

# Set up the window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('gradCPT')

# Load images
city_images = [pygame.image.load(f'labelling/images/city/city_{i}.jpg') for i in range(10)]
mountain_images = [pygame.image.load(f'labelling/images/mountains/mountain_{i}.jpg') for i in range(10)]


def gradCPT():
    clock = pygame.time.Clock()
    target_freq = 0.9
    responses = ([0] * GAME_CYCLES, False * GAME_CYCLES)
    start_timestamp = pygame.time.get_ticks()

    alpha_change_per_frame = 256 / 48

    current_img = np.random.choice(city_images if np.random.rand() < target_freq else mountain_images)

    start_time = pygame.time.get_ticks()

    for i in range(GAME_CYCLES):
        next_img = np.random.choice(city_images if np.random.rand() < target_freq else mountain_images)
        mountain = next_img in mountain_images

        for alpha in np.arange(0, 256, alpha_change_per_frame):
            current_img.set_alpha(255 - alpha)
            next_img.set_alpha(alpha)
            screen.blit(current_img, (0, 0))
            screen.blit(next_img, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    keybr_down = pygame.time.get_ticks() - start_time
                    responses[i] = (keybr_down, mountain)
    
        current_img = next_img

    end_timestamp = pygame.time.get_ticks()
    return start_timestamp, end_timestamp, responses

# if response_time < 800 * 0.7 or response_time > 800 * 1.4:
                        #handle shit
"""
Calculate RTV aka the trial to trial variation in response time
"""
def calculate_RTV(response_times):
    # Z-normalized: normalized such that mean is 0 and std is 1
    z_normalized_RT = (response_times - np.mean(response_times)) / np.std(response_times)
    vtc = np.abs(z_normalized_RT - np.mean(response_times))

    # Linear interpolation for missing values
    # Filling gaps in a simple way, assuming gaps are only one trial wide
    for i in range(1, len(vtc) - 1):
        if np.isnan(vtc[i]):
            vtc[i] = (vtc[i-1] + vtc[i+1]) / 2

    # Smooth the VTC
    VTC_smoothed = gaussian_filter1d(vtc, sigma=7)  # sigma derived from FWHM

    return VTC_smoothed

start_time, end_time, RTs = gradCPT()
RTV = calculate_RTV(RTs)
print(f"Start time: {start_time}")
print(f"End time: {end_time}")
print(f"RTs: {RTs}")
print(f"RTV: {RTV}")