import spacy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import time
from PIL import Image


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
        # print(ent.text, ent.label_)
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

    return counts_df


def main():
    print("Welcome to the MarketWatch Analyzer App!\n")
    print("This app focuses on analyzing MarketWatch articles and is able to count every stock mentioned in the "
          "article as well as perform sentiment analysis on the article.\n")
    while True:
        url = input("Enter a URL to begin analyzing: ")
        print()
        result = article_scraper(url)
        if result is None:
            print("Sorry! It looks like the URL or article is in an unsupported format. Please try another one.")
            continue

        title, body = result[0], result[1]

        # Get title counts
        title_df = get_counts_df(title)
        print("Stocks mentioned in the article title:")
        print(title_df)

        print()
        # Get article counts
        article_body = get_counts_df(body)
        print("Stocks mentioned in the article body:")
        print(article_body)

        print()

        response = input("Would you like to see advanced analyses (sentiment analysis)? Enter [y]/[n]: ")
        print()
        if response in ['y', 'yes', '[y]', 'Y', 'Yes', '[Y]']:
            # Perform sentiment analysis
            title_subjectivity = get_subjectivity(title)
            print("Title subjectivity:", title_subjectivity)

            title_sentiment = get_polarity(title)
            print("Title polarity:", title_sentiment)

            article_subjectivity = get_subjectivity(body)
            print("Article subjectivity:", article_subjectivity)

            article_sentiment = get_polarity(body)
            print("Article polarity:", article_sentiment)

            print()
            print('-----------------------')
            print("Additional information:")
            print("The subjectivity score ranges from 0 to 1, with 0 being objective and 1 being subjective.")
            print("Positive, neutral, and negative polarity scores represent the probability that the sentiment is "
                  "positive, neutral, or negative, respectively. The compound score ranges from -1 to 1, with "
                  "-1 being negative and 1 being positive.")
            print('-----------------------')
            print()

        most_mentioned_company = article_body[article_body.counts == article_body.counts.max()].iloc[0]['company']

        # Request teammate_service
        with open('request.txt', 'w') as file:
            file.write(most_mentioned_company)

        # Wait for teammate_service to run
        while True:
            try:
                with open('response.txt') as file:
                    contents = file.read()
                    break
            except FileNotFoundError:
                time.sleep(1)

        with Image.open(contents) as im:
            im.show()

        response = input("Would you like to analyze another article? Enter [y]/[n]: ")
        if response not in ['y', 'yes', '[y]', 'Y', 'Yes', '[Y]']:
            print("Goodbye!")
            break


if __name__ == '__main__':
    # Setup df and stocks/indexes list
    stops = {'A', 'RBC', 'two'}

    # Data modified from spacy.pythonhumanities.com
    df = pd.read_csv("data/stocks.tsv", sep='\t')

    symbols = df.Symbol.tolist()
    companies = df.CompanyName.tolist()

    # Above DataFrame modified so that Symbol/CompanyName is the index
    # to facilitate efficient lookup
    df_symbol = df.set_index('Symbol')
    df_company_name = df.set_index('CompanyName')

    # Set up spaCy with entity ruler to find stock symbols/companies
    nlp = spacy.blank('en')
    ruler = nlp.add_pipe("entity_ruler")
    patterns = []
    for symbol in symbols:
        if symbol not in stops:
            patterns.append({'label': 'Stock', 'pattern': symbol})
    for company in companies:
        if company not in stops:
            patterns.append({'label': 'Company', 'pattern': company})
    ruler.add_patterns(patterns)

    # Setup vader
    vader = SentimentIntensityAnalyzer()

    main()
