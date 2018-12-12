import logging
import datetime
import time
import os

import yaml
import requests
from bs4 import BeautifulSoup
from cityhash import CityHash64

from ya_parser.database import DB

# todo: write logs to file
logger = logging.getLogger('ya_parser')
logger.setLevel(logging.INFO)

with open(os.path.join(os.getcwd(), 'config.yml')) as yml_file:
    config = yaml.load(yml_file)


class YaParser:
    def __init__(self, verbose=False):
        self.db = DB(os.path.join(os.getcwd(), config['db_name']), create_new=True)
        self.jar = requests.cookies.RequestsCookieJar()
        self.verbose = verbose

    def make_request(self, url):
        response = requests.get(url, cookies=self.jar)
        if not response.ok:
            logger.warning(f'Request error: {response.status_code}')
            return None
        return response.text

    # todo: change html class to story
    def get_section_stories_urls(self, url):
        html = self.make_request(url)
        soup = BeautifulSoup(html, 'html.parser')
        result = []
        for story in soup.find_all(name='a', attrs={'class': 'link link_theme_black i-bem'}, href=True):
            href = story.attrs['href']
            if 'story' in href:
                result.append(config['url_starter'] + href)
        return result

    def get_link_to_all_sources(self, url):
        html = self.make_request(url)
        soup = BeautifulSoup(html, 'html.parser')
        href = soup.find(
            name='a',
            attrs={'class': 'link link_theme_grey story__total i-bem'},
            href=True
        ).attrs['href']
        return config['url_starter'] + href

    # todo: parse site time
    def write_news_titles_to_db(self, url, section):
        result = []
        html = self.make_request(url)
        soup = BeautifulSoup(html, 'html.parser')
        
        if soup.find(name='script', attrs={'src': 'captcha.min.js'}):
            logger.warning('Encountered captcha')
            return
        
        event_title = soup.find(name='h1', attrs={'class': 'story__head'}).text
        event_hash = CityHash64(event_title)
        event_date = datetime.datetime.now().strftime('%Y-%m-%d')

        for story in soup.find_all(name='div', attrs={'class': 'doc doc_for_instory'}):
            title = story.a.text
            title_hash = CityHash64(title)
            agency = story.find(name='div', attrs={'class': 'doc__agency'}).text
            time = story.find(name='div', attrs={'class': 'doc__time'}).text
            if title_hash not in result:
                result.append(
                    dict(
                        event_hash=event_hash, event_title=event_title, event_date=event_date,
                        title_hash=title_hash, title=title, agency=agency, time=time
                    )
                )
        self.db.update_database(section, result)

    def parse_news_section(self, url):
        if self.verbose:
            start_stats = self.db.select_stats()
            start_time = time.time()

        section = url.split('/')[-1].split('.')[0]
        all_stories = self.get_section_stories_urls(url)
        for story in all_stories:
            all_sources = self.get_link_to_all_sources(story)
            self.write_news_titles_to_db(all_sources, section)

        if self.verbose:
            end_time = time.time()
            end_stats = self.db.select_stats()
            logger.info(f'Parsed {section} in {end_time - start_time} seconds\n'
                        f'{end_stats[0] - start_stats[0]} new titles\n'
                        f'{end_stats[1] - start_stats[1]} new events')


if __name__ == '__main__':
    parser = YaParser(verbose=config['verbose'])
    for section in config['url_list']:
        print(section, config['verbose'])
        parser.parse_news_section(section)

