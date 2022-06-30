from flask import Flask, render_template, request, redirect, url_for, flash
import main

app = Flask(__name__)
app.config.from_pyfile("config.py")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze")
def analyze():
    # If user is requesting to analyze a URL
    if request.args.get("url"):
        url = request.args["url"]
        return redirect(url_for("results", url=url))

    # Normal page load
    else:
        news = main.news_scraper()
        return render_template("analyze.html", news=news)


@app.route("/results")
def results():
    url = request.args.get("url")
    # If user attempts to directly go to /results
    if url is None:
        return redirect(url_for("analyze"))
    result = main.article_scraper(url)

    # If the scraper was unsuccessful
    if result is None:
        flash("Oops! It looks like that page is in an unsupported format. "
              "Please try again and ensure that you are entering a MarketWatch article.")
        return redirect(url_for("analyze"))

    title, body = result
    entire_article = title + ". " + body

    counts_table = main.get_counts_df(entire_article)
    subjectivity = main.get_subjectivity(entire_article)
    polarity = main.get_polarity(entire_article)
    polarizing_sentences = main.get_polarizing_sentences(entire_article)

    if counts_table is not None:
        # At least one stock symbol or company is mentioned
        main_company = counts_table.iloc[0]["company"]
        main_symbol = counts_table.iloc[0]["symbol"]
        counts_table = counts_table.to_html(index=False, justify="left",
                                            classes="table table-striped")
    else:
        # No stock symbols or companies are mentioned
        main_company = None
        main_symbol = None

    return render_template("results.html", url=url, title=title,
                           counts_table=counts_table, main_company=main_company,
                           main_symbol=main_symbol, subjectivity=subjectivity,
                           polarity=polarity, polarizing_sentences=polarizing_sentences)


@app.route("/help")
def help():
    return render_template("help.html")


if __name__ == "__main__":
    app.run(debug=True)
