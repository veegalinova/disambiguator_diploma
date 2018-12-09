import logging

import requests
from bs4 import BeautifulSoup
import cityhash

#todo: asyncio
logger = logging.getLogger('ya_parser')


url_starter = 'https://news.yandex.ru/'
URL = 'https://news.yandex.ru/politics.html?from=index'

jar = requests.cookies.RequestsCookieJar()


def get_page_stories_urls(url):
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


def get_news_titles(url):
    response = requests.get(url, cookies=jar)
    if not response.ok:
        logger.warning('resp.getcode() != 200 or resp.geturl() != url')
    soup = BeautifulSoup(response.text, "html.parser")
    for story in soup.find_all(name='div', attrs={'class': 'doc doc_for_instory'}):
        title = story.a.text
        agency = story.div.text
        print(title, agency)


def parse_news_section(section_url):
    stories_urls = get_page_stories_urls(section_url)
    print(stories_urls[0])
    all_sources_pages = get_all_sources_url(stories_urls[0])
    print(all_sources_pages)
    # all_sources_pages = [get_all_sources_url(x) for x in stories_urls]
    # [get_news_titles(x) for x in all_sources_pages]
    get_news_titles(all_sources_pages)


parse_news_section(URL)

