import logging
import datetime
import asyncio
import time

import requests
from bs4 import BeautifulSoup
from cityhash import CityHash64

from ya_parser.database import DB
from ya_parser import DEFAULT_URL_LIST

logger = logging.getLogger('ya_parser')

url_starter = 'https://news.yandex.ru/'
db = DB('db.db', create_new=True)

for URL in DEFAULT_URL_LIST:
    section = URL.split('/')[-1].split('.')[0]

    jar = requests.cookies.RequestsCookieJar()


    def get_section_stories_urls(url):
        result = []
        response = requests.get(url, cookies=jar)
        if not response.ok:
            logger.warning('resp.getcode() != 200 or resp.geturl() != url')
        soup = BeautifulSoup(response.text, "html.parser")

        for story in soup.find_all(name='a', attrs={'class': 'link link_theme_black i-bem'}, href=True):
            href = story.attrs['href']
            if 'story' in href:
                result.append(url_starter + href)

        return result


    def get_all_sources_url(url):
        response = requests.get(url, cookies=jar)
        if not response.ok:
            logger.warning('resp.getcode() != 200 or resp.geturl() != url')
        soup = BeautifulSoup(response.text, "html.parser")
        href = soup.find_all(
                             name='a',
                             attrs={'class': 'link link_theme_grey story__total i-bem'}
                            )[0].attrs['href']
        return url_starter + href


    def write_news_titles(url):
        result = []
        response = requests.get(url, cookies=jar)
        if not response.ok:
            logger.warning('resp.getcode() != 200 or resp.geturl() != url')

        soup = BeautifulSoup(response.text, "html.parser")
        event_title = soup.find(name='h1', attrs={'class': 'story__head'}).text
        event_hash = CityHash64(event_title)
        event_date = datetime.datetime.now().strftime('%Y-%m-%d')

        for story in soup.find_all(name='div', attrs={'class': 'doc doc_for_instory'}):
            title = story.a.text
            agency = story.find(name='div', attrs={'class': 'doc__agency'}).text
            title_hash = CityHash64(title)
            time = story.find(name='div', attrs={'class': 'doc__time'}).text
            if title_hash not in result:
                result.append(dict(event_hash=event_hash, event_title=event_title, event_date=event_date,
                              title_hash=title_hash, title=title, agency=agency, time=time))
        db.update_database(section, result)


    def parse_section(url):
        stories = get_section_stories_urls(url)
        for story in stories:
            source = get_all_sources_url(story)
            write_news_titles(source)


    start = time.time()
    parse_section(URL)
    print(time.time() - start)
