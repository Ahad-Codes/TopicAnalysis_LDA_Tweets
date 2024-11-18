import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Function to generate and save word clouds
def generate_wordclouds(lda_model, num_topics, save_dir="results"):
    # Create the 'results' directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate a word cloud for each topic
    for topic_id in range(num_topics):
        # Extract the top words for the topic
        words = lda_model.show_topic(topic_id, topn=20)  # Top 20 words for each topic
        word_dict = {word: weight for word, weight in words}

        # Sort the words by frequency (or weight in the LDA model)
        sorted_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)

        # Prepare data for generating the word cloud
        words_for_cloud = {word: weight for word, weight in sorted_words}

        # Generate the plot manually
        plot_custom_wordcloud(words_for_cloud, topic_id, save_dir)

def plot_custom_wordcloud(word_dict, topic_id, save_dir):
    # Plot parameters
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    readable_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#f54291", "#228B22",
        "#4169E1", "#FFD700", "#CD5C5C", "#DA70D6", "#8B008B", "#20B2AA",
        "#FF4500", "#A52A2A", "#00CED1", "#9400D3", "#FFA500", "#006400",
        "#FF6347", "#4682B4", "#9ACD32", "#FF1493", "#2F4F4F", "#FF69B4"
    ]

    # Setting plot limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")  # Turn off axes

    # Manually place words on the plot
    positions = []  # Store positions to check for overlap
    max_attempts = 500  # Maximum attempts to place a word
    attempts = 0

    # Try placing each word
    for word, weight in word_dict.items():
        word_size = weight * 1000  # Increase factor to exaggerate size differences
        font_size = max(20, min(int(word_size), 150))  # Adjust the range for more dramatic size

        # Try to place the word in a random position until it fits
        placed = False
        while not placed and attempts < max_attempts:
            attempts += 1
            # Generate a random position within the plot
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)

            # Check for overlap with existing words
            overlap = False
            for pos in positions:
                dist = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
                if dist < 0.05:  # Too close to another word, so overlap
                    overlap = True
                    break

            # If no overlap, place the word
            if not overlap:
                ax.text(x, y, word, fontsize=font_size, ha='center', va='center', color=random.choice(readable_colors), fontweight='bold')
                positions.append((x, y))
                placed = True

    # Save the figure to a file
    plt.title(f"Word Cloud for Topic {topic_id + 1}")
    plt.savefig(f'{save_dir}/wordcloud_topic_{topic_id + 1}.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f'Word Cloud {topic_id + 1} saved successfully to Results folder')
