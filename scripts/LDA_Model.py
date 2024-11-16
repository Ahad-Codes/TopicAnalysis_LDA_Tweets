import gensim
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim
import pickle
import pyLDAvis
import pickle
import pandas as pd
import nltk
from nltk.corpus import words
from tqdm import tqdm

#nltk.download('words')
english_words = set(words.words())

class Model() :
    def __init__(self, df, topics) :
        self.df = df
        self.topics = topics

    def CreateLdaModel(self) :
      
        documents = []
        for sentence in self.df["original_text"]:
            try:
                words = sentence.split()
                true_words = [word for word in words if word.lower() in english_words]
                if true_words:  
                    documents.append(true_words)
                
            except:
                continue

        texts = documents
        self.id2word = corpora.Dictionary(documents)
        self.corpus = [self.id2word.doc2bow(text) for text in texts]

        
        corpus = tqdm(self.corpus, desc="Building Model", total=len(self.corpus))
        self.lda_model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=self.id2word,
            num_topics=self.topics,
            passes=5,
            workers=4  
        )

        print("LDA model built successfully")

    def visualize(self) :
        LDAvis_prepared = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.id2word)
        pyLDAvis.save_html(LDAvis_prepared, r'results\ldavis_prepared_'+ str(self.topics) +'.html')
        print("LDA visualization saved successfully to Results folder")

# if __name__ == '__main__':
#     df = pd.read_csv(r'data\processed\2020_8_dataset.csv')
#     topics = 10
#     model = Model(df,topics)
#     model.CreateLdaModel()
#     model.visualize()
