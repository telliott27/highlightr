import pandas as pd
import gensim
from gensim.models import Doc2Vec,Word2Vec
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from bs4 import BeautifulSoup
import urllib.request
from textstat.textstat import textstat
import re
import time
import numpy as np

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

from multiprocessing import Process, Pool

def remove_html_tags(data):
    """Cleans up text from bs.get_text()"""
    p = re.compile(r'<.*?>')
    dash = re.compile(r'\u200a')
    lb = re.compile(r'\xa0')
    data = p.sub('', data)
    data = dash.sub('-', data)
    data = lb.sub(' ',data)
    return data

def infer_atext(x):
    """Returns the word vector for an article"""
    return(self.a2v.infer_vector(x.words))

def infer_stext(x):
    """Returns the word vector for a sentence"""
    return(self.d2v.infer_vector(x.words))

# create a class for transforming data into features
class createFeatures:
    def __init__(self,pca_comp=20):
        self.columns_to_drop = ['id','sent_id','text','tagged_doc'] # these columns will be used to generate features, but should be dropped prior to training/predicting
        self.pca_comp = pca_comp # save number of PCA components
        self.normalizer = Normalizer() # normalizer to use prior to PCA transformation
        print("class created")

    def fit_transform(self,sentences, sent_vec_space=50, art_vec_space=2 ):
        """Externally facing function to fit the word2vec and PCA to training data and transform the data"""
        X = sentences.reset_index(drop=True) # make sure index is reset to avoid problems with merging later
        self.sent_n = sent_vec_space # length of word vectors of sentences
        self.art_n = art_vec_space # length of word vectors for articles
        arts = X.groupby(by='id')['text'].aggregate(lambda x: ' '.join(x)) # create dataset of entire article to calculate article level features
        articles = pd.DataFrame(arts).reset_index() # reset the index to avoid merging problems later
        self.convertArticles(articles) # convert articles to lists of lists of words
        X = self.convertSentences(X) # convert sentences to lists of lists of sentences
        self.fit_models() # fit doc2vecs for sentence and articles
        print("Data Fit")
        X = self.transform_data(X,articles) # generate features based on passed training data
        Xfeat = X.drop(self.columns_to_drop,axis=1) # drop columns that are not features
        try:
            y = X.highlighted # if highlighted is contained in the data, save it off...
            Xfeat.drop('highlighted',axis=1,inplace=True) # and drop the column from X...
        except:
            y = None # otherwise set y to None
        Xfeat = self.normalizer.fit_transform(Xfeat) # normalize X
        self.pca = PCA(self.pca_comp).fit(Xfeat) # fit PCA to X
        Xfeat = self.pca.transform(Xfeat) # transform X with PCA
        if y is None:
            return(Xfeat) # if no y, just return X
        else:
            return(Xfeat, y) # if there is y, return both X and y

    def fit_models(self):
        """Internal function to fit doc2vecs for sentences and articles"""
        self.sent2vec()
        self.art2vec()

    def transform(self,sentences, return_text = False):
        """Externally facing function to transform data with previously fit featurizers"""
        X = sentences.reset_index(drop=True)
        arts = X.groupby(by='id')['text'].aggregate(lambda x: ' '.join(x))
        articles = pd.DataFrame(arts).reset_index()
        #print(articles)
        self.convertArticles(articles)
        X = self.convertSentences(X)
        X = self.transform_data(X,articles)
        if return_text:
            Xtext = X[['sent_id','text']]
        Xfeat = X.drop(self.columns_to_drop,axis=1)
        try:
            y = X.highlighted
            Xfeat.drop('highlighted',axis=1,inplace=True)
        except:
            y = None
        Xfeat = self.normalizer.transform(Xfeat)
        Xfeat = self.pca.transform(Xfeat)
        if y is None:
            if return_text:
                return(Xfeat,Xtext)
            else:
                return(Xfeat)
        else:
            if return_text:
                return(Xfeat, y, Xtext)
            else:
                return(Xfeat, y)

    def transform_no_pca(self, sentences):
        """Externally facing function to transform data but not pass through PCA, for analyzing features"""
        X = sentences.reset_index(drop=True)
        arts = X.groupby(by='id')['text'].aggregate(lambda x: ' '.join(x))
        articles = pd.DataFrame(arts).reset_index()
        #print(articles)
        self.convertArticles(articles)
        X = self.convertSentences(X)
        X = self.transform_data(X,articles)
        return(X)

    def transform_data(self,X,articles):
        """Internal function to transform the data"""
        # save off location of sentence in article
        sent_num = X[['sent_id','id','sent_num']]
        # calculate length of each sentence in words
        num_words = X[['sent_id']]
        num_words['num_words'] = [len(s.split()) for s in X.text]
        # generate doc2vec vectors for articles
        arts_vec = self.getArtVec()
        col_names = ['art'+str(i) for i in range(self.art_n)]
        arts_vec = pd.concat([articles,arts_vec],axis=1)[['id']+col_names]
        # calculate reading level of article
        read_lev = self.artReadingLevel(articles)
        # generate doc2vec vectors for sentences
        sent_vecs = self.getSentVec(X[['sent_id','tagged_doc']])
        # merge on number of words
        X = pd.merge(X,num_words,on='sent_id')
        # merge on article vectors
        X = pd.merge(X,arts_vec,on='id')
        # merge on reading level
        X = pd.merge(X,read_lev,on='id')
        # merge on sentence vectors
        X = pd.merge(X,sent_vecs,on='sent_id')
        return(X)

    def convertArticles(self,articles):
        """Converts articles to lists of lists of words"""
        self.arts = []
        for i, row in articles.iterrows():
            this_doc = gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row['text'],min_len=1), [i])
            self.arts.append(this_doc)

    def infer_atext(self,x):
        """Get dov2vec vectors for article"""
        return(self.a2v.infer_vector(x.words))

    def infer_stext(self,x):
        """Get doc2vec vectors for sentence"""
        return(self.d2v.infer_vector(x.words))

    def art2vec(self):
        """Trains doc2vec on articles"""
        self.a2v = Doc2Vec(self.arts,size=self.art_n,window=8,min_count=5,workers=4)

    def getArtVec(self):
        vecs = [self.a2v.infer_vector(d.words) for d in self.arts] #convert articles to vectors
        col_names = ['art'+str(i) for i in range(self.art_n)] # generate new column names
        arts_vec = pd.DataFrame(vecs,columns=col_names) #convert vectors to dataframe
        return(arts_vec)

    def artReadingLevel(self,articles):
        rldf = articles[['id']]
        pool = Pool(processes=1)
        rldf['reading_level'] = pool.map(textstat.flesch_kincaid_grade,articles.text)
        pool.close()
        return(rldf)

    def convertSentences(self,X):
        X['tagged_doc'] = X.apply(lambda x: gensim.models.doc2vec.TaggedDocument(
            gensim.utils.simple_preprocess(x['text'],min_len=1), [x['sent_id']]), axis=1)
        self.sents = X.tagged_doc.tolist()
        return(X)

    def sent2vec(self):
        self.d2v = Doc2Vec(self.sents, size=self.sent_n, window=8, min_count=5, workers=4)

    def getSentVec(self,df):
        vecs = [self.infer_stext(x) for x in df.tagged_doc]
        col_names = ['sen'+str(i) for i in range(self.sent_n)]
        vecs2 = pd.DataFrame(vecs,columns=col_names, index=range(len(vecs)))
        vecs2['rownum'] = vecs2.index
        df2 = df.drop('tagged_doc',axis=1)
        df2['rownum'] = df2.index
        df2 = pd.merge(df2,vecs2,on='rownum')[['sent_id']+col_names]
        return(df2)

    def getSentiment(self,df):
        sic = SentimentIntensityAnalyzer()
        def return_comp(x):
            return(sic.polarity_scores(x)['compound'])
        df2 = pd.DataFrame({'sent_id': df['sent_id'],
                'sentiment': [return_comp(x) for x in df.text] })
        return(df2)


class runModel(createFeatures): # contains the model that will be used to predict highlights, inherets from createFeatures
    def __init__(self,model,train, sm=None, sent_vec_space=50, art_vec_space=50, pca_n=20):
        """
        model = sklearn classifier
        train = training data
        sm = SMOTE object if want to balance training data
        sent_vec_space = length of sentence vectors
        art_vec_space = length of article vectors
        pca_n = number of PCA components
        """
        createFeatures.__init__(self, pca_comp=pca_n)
        self.model = model
        self.sent_v = sent_vec_space
        self.art_v = art_vec_space
        self.pca_n = pca_n
        self.Xtrain, self.ytrain = self.fit_transform(train, sent_vec_space=self.sent_v, art_vec_space=self.art_v)
        self.sm = sm
        if self.sm is not None:
            self.Xtrain, self.ytrain = self.sm.fit_sample(self.Xtrain, self.ytrain)
        self.model.fit(self.Xtrain,self.ytrain)
        print("Model Fit.")
        del self.Xtrain
        del self.ytrain

    def getScores(self,test):
        """Get validation scores given held-out test data"""
        Xtest, ytest = self.transform(test)
        #if self.sm is not None:
        #    Xtest, ytest = self.sm.fit_sample(Xtest,ytest)
        predicted = self.model.predict(Xtest)
        print("Accuracy {:.4}".format(metrics.accuracy_score(ytest, predicted)))
        print(metrics.classification_report(ytest, predicted,target_names=['Not Highlighted','Highlighted']))
        return Xtest, ytest, predicted

    def getTopHighlights(self,sentences):
        """given a data frame of sentences, returns the top five most likely highlights"""
        X, Xtext = self.transform(sentences, return_text=True)
        probs = self.model.predict_proba(X) #get predicted probabilities
        df = pd.DataFrame({'text': Xtext.text, 'prob': probs[:,1], 'sent_id': Xtext.sent_id})
        cutoff = np.percentile(df['prob'],80) #get the cutoff for probabilities in the top 20%
        best_high = df[df.prob>cutoff].reset_index(drop=True)
        best_high['diff'] = best_high['sent_id'].diff()
        for i in best_high.index[:0:-1]:
            # for the sentences in the top 20%, combine contiguous sentences
            if best_high['diff'][i] == 1:
                best_high.loc[i-1,'text'] = best_high.loc[i-1,'text'] + ' ' + best_high.loc[i,'text']
                best_high.loc[i-1,'prob'] = max([best_high.loc[i-1,'prob'],best_high.loc[i,'prob']])
                best_high.drop(i,axis=0,inplace=True)
        df = best_high.sort_values('prob',ascending=False).iloc[0:5,] # return top five passages
        return(df)

    def getArticleFromUrl(self,url):
        """Grab article from medium.com url"""
        class AppURLopener(urllib.request.FancyURLopener):
            version = "Mozilla/5.0"
        opener = AppURLopener()
        try: # try to find the tag containing the article text
            response = opener.open(url)
            soup = BeautifulSoup(response,"lxml")
            article = ''
            article = soup.find('article').find('div',class_='postArticle-content').get_text(' ')
        except: # if no tag containing article text, skip
            return(None, None)
        article = remove_html_tags(article)
        return(article, soup)

    def cleanArticle(self,article, soup):
        """Given article from medium.com URL, clean up the text"""
        punct = re.compile(r'[.?!,":;]')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        article_pd = pd.DataFrame({'text': [article], 'id': [1]})
        # convert article text to list of sentences
        all_sentences = [tokenizer.tokenize(s) for s in article_pd.text]
        article_pd['sentences'] = all_sentences
        df = article_pd.set_index(['id'])['sentences'].apply(pd.Series).stack().reset_index()
        df.columns = ['id','sent_num','text']
        df['sent_id'] = range(len(df.id))
        def ainb(s,hs):
            if any([s.find(h) != -1 for h in hs]):
                return(1)
            else:
                return(0)
        blocks = []
        for b in soup.find_all("blockquote"):
            blocks += [punct.sub('',remove_html_tags(k)) for k in tokenizer.tokenize(b.get_text(' '))]
        text = [punct.sub('',s).strip() for s in df['text']]
        df['blockquote'] = [ainb(s,blocks) for s in text]
        return(df)

    def topHighlightsFromText(self,text):
        punct = re.compile(r'[.?!,":;]')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        article_pd = pd.DataFrame({'text': [text], 'id': [1]})
        all_sentences = [tokenizer.tokenize(s) for s in article_pd.text]
        article_pd['sentences'] = all_sentences
        df = article_pd.set_index(['id'])['sentences'].apply(pd.Series).stack().reset_index()
        df.columns = ['id','sent_num','text']
        df['sent_id'] = range(len(df.id))
        df['blockquote'] = 0
        top_high = self.getTopHighlights(df)
        return(top_high)

    def topHighlightsFromURL(self,url, getTitle=False):
        article, soup = self.getArticleFromUrl(url)
        if article is None:
            return(None, None)
        df = self.cleanArticle(article,soup)
        top_high = self.getTopHighlights(df)
        if getTitle:
            title = soup.find('h1').get_text(' ')
            return(top_high, title)
        else:
            return(top_high)
