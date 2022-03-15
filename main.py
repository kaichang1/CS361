import spacy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup


def news_scraper():
    news = []
    url = "https://www.marketwatch.com/investing?mod=top_nav"

    try:
        webpage = requests.get(url)
    except Exception as err:
        print("Error!")
        print(f"Message: {err}")
        print(f"Type: {type(err)}")
        return None

    soup = BeautifulSoup(webpage.content, 'html.parser')
    articles = soup.find(attrs={'class': 'collection__elements j-scrollElement'})
    articles = articles.find_all(attrs={'class': 'element--article'})

    for article in articles:
        link = article.find(attrs={'class': 'link'})
        if link:
            news.append(link)

    return news


def article_scraper(url):
    try:
        webpage = requests.get(url)
    except requests.exceptions.MissingSchema:
        return None

    soup = BeautifulSoup(webpage.content, 'html.parser')
    try:
        title = soup.find(attrs={'itemprop': 'headline'}).get_text()
        body = soup.find(attrs={'itemprop': 'articleBody'}).get_text()
    except AttributeError:
        return None

    return title, body


# Get subjectivity score
# Output range within [0.0, 1.0], where 0.0 is objective and 1.0 is subjective
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Get polarity scores
# Output:
#   pos: probability that the sentiment is positive
#   neu: probability that the sentiment is neural
#   neg: probability that the sentiment is negative
#   compound: normalized compound score, range within [-1, 1], where -1 is negative and 1 is positive
def get_polarity(text):
    return vader.polarity_scores(text)


# Get a DataFrame that counts the number of times a stock was mentioned
# columns are: company, symbol, counts
def get_counts_df(text):
    counts = []
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == 'Stock':
            counts.append(df_symbol.loc[ent.text]['CompanyName'])
        else:
            counts.append(ent.text)

    counts_df = pd.DataFrame(counts).reset_index()
    try:
        counts_df = counts_df.groupby(0).index.count().reset_index().rename(columns={0: 'company', 'index': 'counts'})
    except KeyError:
        return None

    counts_df['symbol'] = counts_df.company.apply(lambda x: df_company_name.loc[x]['Symbol'])
    counts_df = counts_df[['company', 'symbol', 'counts']]
    counts_df = counts_df.sort_values(by='counts', ascending=False)

    return counts_df


# Get the sentences that are rated as most negative and most positive.
def get_polarizing_sentences(text):
    doc = nlp(text)

    min_polarity = float('inf')
    max_polarity = float('-inf')
    min_sentence = None
    max_sentence = None

    for sent in doc.sents:
        polarity = get_polarity(sent.text)['compound']
        if polarity < min_polarity:
            min_polarity = polarity
            min_sentence = sent
        if polarity > max_polarity:
            max_polarity = polarity
            max_sentence = sent
        
    return min_sentence, max_sentence


### Setup ###

# Stop words
stops = {'A', 'RBC', 'two', 'UK'}

# stocks.tsv modified from spacy.pythonhumanities.com
df = pd.read_csv("static/data/stocks.tsv", sep='\t')

symbols = df.Symbol.tolist()
companies = df.CompanyName.tolist()

# DataFrame modified so that Symbol/CompanyName is the index to facilitate efficient lookup
df_symbol = df.set_index('Symbol')
df_company_name = df.set_index('CompanyName')

df_symbol = df_symbol[['CompanyName']]
# Some companies have multiple stock symbols which results in rows with duplicate indices.
# To handle this, we group by company name and set the `symbol` column to equal all grouped stock symbols separated by commas
# https://stackoverflow.com/questions/50422809/pandas-group-by-with-all-the-values-of-the-group-as-comma-separated
df_company_name = df_company_name.groupby(df_company_name.index).Symbol.agg([('Symbol', ', '.join)])

# Set up spaCy with sentencizer and entity ruler to find stock symbols/companies
nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')
ruler = nlp.add_pipe("entity_ruler")
patterns = []
for symbol in symbols:
    if symbol not in stops:
        patterns.append({'label': 'Stock', 'pattern': symbol})
for company in companies:
    if company not in stops:
        patterns.append({'label': 'Company', 'pattern': company})
ruler.add_patterns(patterns)

# Set up vader
vader = SentimentIntensityAnalyzer()
