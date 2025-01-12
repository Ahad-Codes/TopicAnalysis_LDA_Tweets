# COVID-19 Topic Analysis using Latent Dirichlet Allocation (LDA) and Sentiment Analysis 🦠💬

## Motivation 🌍

The COVID-19 pandemic has had an unparalleled impact on global society, influencing public discourse on social media platforms. This project aims to analyze public sentiment and the evolving topics of discussion on Twitter related to the pandemic. By utilizing Latent Dirichlet Allocation (LDA) for topic modeling and the VADER sentiment analysis tool, we seek to uncover insights into the emotional dynamics and focus of public discussions at various stages of the pandemic.

The goal of this analysis is to explore how people's concerns and emotions evolved over time and to provide insights into the public's focus on critical issues like personal safety, economic impacts, healthcare, and government responses.

## Project Goals 🎯

- **Topic Extraction**: Use LDA to uncover hidden topics from a large dataset of pandemic-related tweets.
- **Sentiment Analysis**: Utilize VADER sentiment analysis to categorize public sentiment into positive, negative, and neutral categories.
- **Public Attention and Emotional Trends**: Examine how public focus and emotional responses changed throughout different phases of the pandemic.
- **Insights for Public Health Communication**: Provide actionable insights for policymakers and health organizations about the emotional and thematic dynamics of public discourse.

## Implementation 🛠️

### Data Collection 📊

The dataset consists of over 1 million tweets related to COVID-19, spanning three periods:
- April 19 to June 20, 2020 🗓️
- August 20 to October 20, 2020 🗓️
- April 26 to June 27, 2021 🗓️

Each tweet was preprocessed to clean the text by removing irrelevant content such as URLs, hashtags, and retweet markers. We then removed stop words and tweets with fewer than three words to ensure a clean dataset for analysis.

### Topic Modeling with LDA 📑

We applied the Latent Dirichlet Allocation (LDA) model to the preprocessed tweet data to identify the main topics of discussion. LDA is an unsupervised machine learning model that discovers hidden thematic structures within a collection of documents (in our case, tweets). The LDA model was trained using Gibbs sampling for topic assignment, leveraging the `gensim` library for implementation.

Four primary topics were identified:
1. Personal Impact and Safety 🛡️
2. Pandemic Updates and Statistics 📈
3. Economic and Global Impacts 💸
4. Community Health and Testing 🏥

We visualized these topics using word clouds and an inter-topic distance map to understand the relationships between them.

### Sentiment Analysis with VADER 💬

For sentiment analysis, we used VADER (Valence Aware Dictionary and sEntiment Reasoner), which is a rule-based sentiment analysis tool. It classifies each tweet into positive, negative, or neutral sentiment categories. Sentiment analysis was applied to understand how public emotions changed over time during the pandemic.

### Results 📊

- **LDA Topics**: Four distinct topics were identified from the tweets, each reflecting significant aspects of public discourse during the pandemic.
- **Sentiment Trends**: Overall, tweets exhibited a shift towards more positive sentiment as the pandemic progressed, especially in 2021 with the rollout of vaccines 💉.
- **Word Clouds**: Each sentiment category (positive, neutral, negative) was analyzed to generate word clouds that helped visualize the dominant themes associated with public discourse.
- **Sentiment Over Time**: A temporal analysis revealed that public sentiment fluctuated significantly, with key events causing noticeable spikes in sentiment.

### Challenges and Limitations ⚠️

- The LDA model's performance was influenced by the preprocessing steps, which may have removed some relevant terms or introduced noise.
- VADER, while efficient, struggled to capture the nuanced sentiment in sarcastic or ambiguous tweets, potentially misclassifying some content.
- The dataset may not represent all geographic regions or demographics equally, limiting the generalizability of the findings.

## Future Directions 🚀

- Improve the data preprocessing pipeline to handle edge cases such as sarcastic tweets more effectively.
- Explore the use of advanced natural language processing models (e.g., BERT, GPT) for more accurate sentiment classification.
- Expand the analysis to incorporate additional data such as geospatial information, policy changes, and news events to better understand the broader context of public discourse.



