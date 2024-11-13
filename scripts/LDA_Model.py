import gensim
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim
import pickle
import pyLDAvis
import pickle
import pandas as pd

class Model() :
    def __init__(self, path, topics) :
        self.path = path
        self.topics = topics

    def CreateLdaModel(self) :
        df = pd.read_csv(self.path, encoding='utf-8')

        documents = []
        for sentence in df["original_text"]:
            try:
                words = sentence.split()
                documents.append(words)  # Append the list of words for each sentence
            except:
                continue

        texts = documents
        self.id2word = corpora.Dictionary(documents)
        self.corpus = [self.id2word.doc2bow(text) for text in texts]

        # Build LDA model
        self.lda_model = gensim.models.LdaMulticore(corpus=self.corpus, id2word=self.id2word, num_topics=self.topics, passes=5)
        #pprint(lda_model.print_topics())
        #doc_lda = lda_model[corpus]

    def visualize(self) :
        LDAvis_prepared = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.id2word)
        pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(self.topics) +'.html')
        print(LDAvis_prepared)

if __name__ == '__main__':
    path = r'content/data_processed_2021_4_dataset.csv'
    topics = 10
    model = Model(path,topics)
    model.CreateLdaModel()
    model.visualize()
