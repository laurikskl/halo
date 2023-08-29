import pandas as pd
import numpy as np
from scipy.stats import skew
import mne
import matplotlib.pyplot as plt


# Adds previous 9 windows from the feature_df to current window.
# For first 9 windows, 0 values are used as placeholders
def add_windows(feature_df, num_windows=9):
    new_features = []

    for i in range(len(feature_df)):
        current_window = feature_df.iloc[i]
        if i < num_windows:
            previous_windows = pd.DataFrame(index=range(num_windows - i), columns=feature_df.columns)
            previous_windows = previous_windows.fillna(0)
            previous_windows = pd.concat([previous_windows, feature_df.iloc[:i]])
        else:
            previous_windows = feature_df.iloc[i - num_windows:i]
        concatenated_features = pd.concat([current_window, previous_windows.unstack()]).reset_index(drop=True)
        new_features.append(concatenated_features)
    
    return pd.DataFrame(new_features)
    
# Returns a DataFrame with the features extracted from the data

# Approx. Entropy  Total variation  Standard variation      Energy  Skewness
# 0.000637         0.253894          0.077180             308.528797  0.280371
def feature_extraction(data, sequence_length = 0.8, hz = 256):
    # Create empty lists to store our results
    approx_entropy = []
    total_variation = []
    standard_variation = []
    energy = []
    sample_skewness = []

    segment_size = int(sequence_length * hz)   # 80% of 256 Hz = 204.8 Hz

    for i in range(0, len(data), segment_size):
        segment = data['Delta_TP9'].iloc[i:i+segment_size]
        
        if len(segment) < segment_size: break
        segment = segment.reset_index(drop=True)

        # print(segment)
        # Approximate Entropy
        # entropy = approximate_entropy(segment)
        approx_entropy.append(approximate_entropy(segment))
        
        # Total Variation
        total_variation.append(np.sum(np.gradient(segment)))
        
        # Standard Variation (Standard Deviation)
        standard_variation.append(np.std(segment))
        
        # Energy
        energy.append(np.sum(segment**2))
        
        # Skewness
        sample_skewness.append(skew(segment))

    # You can now convert these lists into a DataFrame
    features_df = pd.DataFrame({
        'Approx. Entropy': approx_entropy,
        'Total variation': total_variation,
        'Standard variation': standard_variation,
        'Energy': energy,
        'Skewness': sample_skewness
    })

def approximate_entropy(segment, m=2):
    r = 0.2 * np.std(segment)
    
    phi_values = []
    for m in range(2):
        inner_phi = np.mean([np.exp(np.mean(np.abs(segment[j] - segment[j+m]))) for j in range(len(segment) - m)])
        phi_values.append(inner_phi)
    
    phi = np.mean(phi_values)
    return np.log(phi)