import spacy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup


def news_scraper():
    """Scrape a list of news articles from MarketWatch.

    The news is taken from: https://www.marketwatch.com/investing?mod=top_nav

    Returns:
        list: list of html <a> tags of type bs4.element.Tag.
            Each element represents a single news article.
    """
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
    """Scrape the text from a MarketWatch article.

    The scraped article text is divided into title and body segments.

    Args:
        url (string): article URL
    Returns:
        tuple: tuple consisting of (title, body), where title and body are strings.
    """
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


def get_subjectivity(text):
    """Get subjectivity score.

    The output range is [0.0, 1.0], where 0.0 is objective and 1.0 is subjective.

    Args:
        text (string): text to analyze
    Returns:
        float: subjectivity score
    """
    return TextBlob(text).sentiment.subjectivity


def get_polarity(text):
    """Get polarity scores.

    Output:
        pos: probability that the sentiment is positive
        neu: probability that the sentiment is neural
        neg: probability that the sentiment is negative
        compound: normalized compound score, range is [-1.0, 1.0],
            where -1.0 is negative and 1.0 is positive
    
    Args:
        text (string): text to analyze
    Returns:
        dict: dictionary of polarity scores.
            Keys are 'pos', 'neu', 'neg', and 'compound'.
            Values are floats representing the associated positive, neutral,
                negative, and compound polarity scores.
    """
    return vader.polarity_scores(text)


def get_polarizing_sentences(text):
    """Get the sentences rated as most negative and most positive.
    
    Args:
        text (string): text to analyze
    Returns:
        tuple: tuple consisting of (most negative sentence, most positive sentence).
            Each tuple element is of type spacy.tokens.span.Span.
    """
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


def get_counts_df(text):
    """Count the number of times each stock or company is mentioned.

    Args:
        text (string): text to analyze
    Returns:
        DataFrame: DataFrame consisting of company, symbol, and counts columns.
            Each row is a unique company that is detected in the text.
    """
    counts = []
    doc = nlp(text)

    # Iterate through doc entities and if the entity is a Stock, then we find
    # the corresponding CompanyName of that Stock and append it to our counts
    # list. If the entity is a CompanyName, then we simply append it to our counts list.
    for ent in doc.ents:
        if ent.label_ == 'Stock':
            counts.append(df_symbol.loc[ent.text]['CompanyName'])
        else:
            counts.append(ent.text)

    # Convert company counts list into a DataFrame. This gives us a DataFrame
    # with company and counts columns.
    counts_df = pd.DataFrame(counts).reset_index()
    try:
        counts_df = counts_df.groupby(0).index.count().reset_index().rename(
            columns={0: 'company', 'index': 'counts'})
    except KeyError:
        return None

    # Add a symbol column to the DataFrame and sort by counts.
    counts_df['symbol'] = counts_df.company.apply(lambda x: df_company_name.loc[x]['Symbol'])
    counts_df = counts_df[['company', 'symbol', 'counts']]
    counts_df = counts_df.sort_values(by='counts', ascending=False)

    return counts_df


### Setup ###

# Stop words
stops = {'A', 'RBC', 'two', 'UK'}

# stocks.tsv modified from spacy.pythonhumanities.com
df = pd.read_csv("static/data/stocks.tsv", sep='\t')

# List of stock symbols and companies.
# Used to create a patterns list for the entity ruler.
symbols = df.Symbol.tolist()
companies = df.CompanyName.tolist()

# DataFrame modified so that Symbol/CompanyName is the index to facilitate efficient lookup.
# Used in get_counts_df()
df_symbol = df.set_index('Symbol')
df_company_name = df.set_index('CompanyName')

df_symbol = df_symbol[['CompanyName']]
# Some companies have multiple stock symbols which results in rows with
# duplicate indices. To handle this, we group by company name and set the
# `symbol` column to equal all grouped stock symbols separated by commas
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
