import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import random

# Load preprocessed CSV data
data = pd.read_csv('data/2020_8_dataset.csv')

data['original_text'] = data['original_text'].fillna('')

# Extract words and compute frequencies
all_words = []

for tweet in data['original_text']:
    words = tweet.split()
    all_words.extend(words)

# Exclude the word "covid19" from the list of words
all_words = [word for word in all_words if word.lower() != "covid19"]

word_freq = Counter(all_words)
top_words = word_freq.most_common(25)

# Normalize word frequencies to scale font sizes
max_font_size = 100  # Maximum font size for the most frequent word
min_font_size = 10  # Minimum font size
max_freq = top_words[0][1]
min_freq = top_words[-1][1]

# Print out the frequency of each word
for word, freq in top_words:
    print(f"Word: '{word}', Frequency: {freq}")

# Adjust font sizes to make size differences more noticeable
max_font_size = 100  # Maximum font size
min_font_size = 8    # Minimum font size
max_freq = top_words[0][1]  # Frequency of the most common word
min_freq = top_words[-1][1]  # Frequency of the 25th most common word

# Scaling function to determine font size
def scale_font_size(freq):
    return int((freq - min_freq) / (max_freq - min_freq) * (max_font_size - min_font_size) + min_font_size)

# Initialize plot
plt.figure(figsize=(12, 6))
plt.axis('off')

# Set plotting area dimensions
width, height = 800, 400
plt.xlim(0, width)
plt.ylim(0, height)

# Track used coordinates and bounding boxes for words to avoid overlap
used_positions = []

# Define a palette of readable colors
readable_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#f54291", "#228B22",
    "#4169E1", "#FFD700", "#CD5C5C", "#DA70D6", "#8B008B", "#20B2AA",
    "#FF4500", "#A52A2A", "#00CED1", "#9400D3", "#FFA500", "#006400",
    "#FF6347", "#4682B4", "#9ACD32", "#FF1493", "#2F4F4F", "#FF69B4"
]

# Function to generate a random color
def random_color():
    return random.choice(readable_colors)

# Function to check if the word's bounding box overlaps with any used positions
def check_overlap(x, y, word, font_size):
    for px, py, p_font_size in used_positions:
        # Calculate the distance between word centers
        distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        # Calculate the minimum distance based on font size
        min_distance = font_size * len(word) + p_font_size
        if distance < min_distance:
            return True
    return False

# Place each of the top 25 words on the plot
for word, freq in top_words:
    font_size = scale_font_size(freq)
    placed = False
    attempt_count = 0
    
    while not placed and attempt_count < 100:  # Try up to 100 times to find a non-overlapping position
        x = random.randint(0, width - font_size * len(word))
        y = random.randint(0, height - font_size)
        
        # Check if the new position overlaps with any previously placed words
        if not check_overlap(x, y, word, font_size):
            used_positions.append((x, y, font_size))
            plt.text(x, y, word, fontsize=font_size, color=random_color(), ha='left', va='bottom')
            placed = True
        
        attempt_count += 1

    # If after 100 attempts the word could not be placed, place it anyway
    if not placed:
        x = random.randint(0, width - font_size * len(word))
        y = random.randint(0, height - font_size)
        plt.text(x, y, word, fontsize=font_size, color=random_color(), ha='left', va='bottom')
        print(f"Placed word '{word}' with overlap.")

plt.show()