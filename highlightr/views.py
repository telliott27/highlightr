from flask import render_template, request
from highlightr import app
from pipeline import runModel

import pandas as pd
import numpy as np
import pickle

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
from textstat.textstat import textstat

from bs4 import BeautifulSoup
import urllib.request

from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE


the_model = pickle.load(open('model.p','rb'))



@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Highlightr',
       )


@app.route('/results')
def results():
  url = request.args.get('url')
  top_high, title = the_model.topHighlightsFromURL(url, getTitle=True)
  if top_high is None:
      return render_template('oops.html')
  top_high = top_high.to_dict('records')

  return render_template("results.html", highlights = top_high, art_title = title)

@app.route('/text', methods=['GET','POST'])
def text():
    text = request.form['article_text']
    top_high = the_model.topHighlightsFromText(text)
    top_high = top_high.to_dict('records')
    return render_template("results.html", highlights=top_high, art_title='Entered Text')
