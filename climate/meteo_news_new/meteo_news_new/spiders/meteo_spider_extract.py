import scrapy
import warnings

from scrapy.exceptions import ScrapyDeprecationWarning
warnings.simplefilter('ignore', ScrapyDeprecationWarning)

class MeteoSpider(scrapy.Spider):
    name = 'MeteoSpider'
    allowed_domains = ['stirileprotv.ro']
    base_url = 'https://stirileprotv.ro/stiri/vremea/?page='
    start_page = 2  # Initialize the page counter

    custom_settings = {
        'FEEDS': {
            'meteo_news.json': {
                'format': 'json',
                'encoding': 'utf8',
                'overwrite': True
            }
        },
        'RETRY_HTTP_CODES': [429, 500, 502, 503, 504, 522, 524, 408],
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36',
        'ROBOTSTXT_OBEY': False,
        # 'DOWNLOAD_DELAY': 10,
        'RETRY_TIMES': 20
    }

    def start_requests(self):
        # Start with the initial page
        yield scrapy.Request(self.base_url + str(self.start_page), self.parse)

    def handle_failure(self, failure):
        self.logger.error(f"Request failed for URL: {failure.request.url}")
        # After processing the page, increment the page counter
        self.start_page += 1
        # Proceed to the next page, even if the current page failed
        if self.start_page <= 5:
            yield scrapy.Request(self.base_url + str(self.start_page), self.parse, errback=self.handle_failure)

    def parse(self, response):
        # Extracting the headers of all news articles
        for article in response.xpath('//article[contains(@class, "grid article")]'):
            title = article.xpath('.//div[@class="article-title"]/h2/a/text()').get()
            date = article.xpath('.//div[@class="article-date"]/text()').get()
            article_url = article.css('div.article-title h2 a::attr(href)').get()

            # Store the extracted data only if title is present
            if article_url:
                yield {
                    'article_title': title.strip(),
                    'article_date': date.strip() if date else None,
                    'article_url': response.urljoin(article_url) if article_url else None
                }

        # After processing the page, increment the page counter
        self.start_page += 1

        # If you only want to scrape a specific number of pages (e.g., the first 5):
        if self.start_page <= 12:
            yield scrapy.Request(self.base_url + str(self.start_page), self.parse, errback=self.handle_failure)

