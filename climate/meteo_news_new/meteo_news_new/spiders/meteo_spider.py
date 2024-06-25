import scrapy
import json
import warnings
from scrapy.exceptions import ScrapyDeprecationWarning

warnings.simplefilter('ignore', ScrapyDeprecationWarning)


class MeteoSpiderIndividual(scrapy.Spider):
    name = 'MeteoSpiderIndividual'
    start_urls = []

    # Read the URLs from the previously generated JSON file
    with open('meteo_news.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        start_urls = [article['article_url'] for article in data]

    custom_settings = {
        'FEEDS': {
            'meteo_news_updated.json': {
                'format': 'json',
                'encoding': 'utf8',
                'overwrite': True
            }
        },
        'RETRY_HTTP_CODES': [429, 500, 502, 503, 504, 522, 524, 408],
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36',
        'ROBOTSTXT_OBEY': False,
        # 'DOWNLOAD_DELAY': 1,
        'RETRY_TIMES': 20
    }

    def parse(self, response):
        # Extract the article lead
        article_lead = response.css('div.article--lead p::text').get()

        # Extract headers and paragraphs from the article text
        article_headers = response.css(
            'div.article--text h1::text, div.article--text h2::text, div.article--text h3::text, div.article--text p::text').getall()

        # Extract article title
        article_title = response.css('div.article--title h1::text').get()

        # Extract article date (this is based on the structure of the original spider, adjust as needed)
        article_date_element = response.css('div.article--published-date::text').getall()
        article_date = ' '.join([date.strip() for date in article_date_element if
                                 date.strip()])  # This removes extra spaces and combines the date and time

        yield {
            'article_url': response.url,
            'article_title': article_title,
            'article_lead': article_lead,
            'article_text': article_headers,
            'article_date': article_date
        }