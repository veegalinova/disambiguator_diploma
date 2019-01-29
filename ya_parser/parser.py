import logging
import datetime
import time
import os
import sqlite3

import yaml
import requests
from bs4 import BeautifulSoup
from cityhash import CityHash64

# from disambiguator.ya_parser.database import DB

# todo: write logs to file
logger = logging.getLogger('ya_parser')
logger.setLevel(logging.INFO)

with open(os.path.join(os.getcwd(), 'ya_parser/config.yml')) as yml_file:
    config = yaml.load(yml_file)


class DB:
    def __init__(self, db_file, create_new=True):
        self.conn = self.create_conn(db_file)
        if create_new:
            self.create_tables()
        self.cur = self.conn.cursor()

    @staticmethod
    def create_conn(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except:
            logger.error(f'Database connection error')
        return conn

    def create_tables(self):
        tables = {'events': ' (hash INT PRIMARY KEY, title TEXT, section TEXT, date DATE, is_clustered INT)',
                  'news_titles': ' (hash INT PRIMARY KEY, title TEXT, time TEXT, event INT, source TEXT)'}
        try:
            for table_name, columns in tables.items():
                sql = 'CREATE TABLE IF NOT EXISTS ' + table_name + columns
                self.conn.execute(sql)
        except Exception as ex:
            logger.error(f'Table creation error {ex}')

    def insert_event(self, event):
        values = f'{event[0]}, \'{event[1]}\', \'{event[2]}\', \'{event[3]}\''
        query = f'INSERT INTO events(hash, title, section, date) ' \
                f'SELECT {values} ' \
                f'WHERE NOT EXISTS (SELECT 1 FROM events WHERE hash={event[0]})'
        self.conn.execute(query)

    def insert_title(self, title):
        values = f'{title[0]}, \'{title[1]}\', \'{title[2]}\', \'{title[3]}\', \'{title[4]}\' '
        query = f'INSERT INTO news_titles(hash, title, time, event, source) ' \
                f'SELECT {values} ' \
                f'WHERE NOT EXISTS (SELECT 1 FROM news_titles WHERE hash={title[0]})'
        self.conn.execute(query)

    def update_database(self, section, data):
        self.conn.execute('BEGIN TRANSACTION;')
        for line in data:
            self.insert_event((line['event_hash'], line['event_title'], section, line['event_date']))
            self.insert_title((line['title_hash'], line['title'], line['time'], line['event_hash'], line['agency']))
        self.conn.execute('COMMIT;')

    def _insert_titles_from_events(self, events):
        self.conn.execute('BEGIN TRANSACTION;')
        for event in events:
            title = (event[0], event[1], 'null', event[0], 'Yandex')
            self.insert_title(title)
        self.conn.execute('COMMIT;')

    def merge_events(self, events):
        self._insert_titles_from_events(events)
        query1 = f'DELETE ' \
                 f'FROM events ' \
                 f'WHERE hash={events[1][0]} '
        query2 = f'UPDATE news_titles ' \
                 f'SET event = {events[0][0]} ' \
                 f'WHERE event = {events[1][0]} '
        print(query1)
        self.cur.execute(query1)
        self.cur.execute(query2)

    def select_stats(self):
        self.cur.execute('SELECT count(*) FROM news_titles UNION ALL SELECT count(*) FROM events')
        return [x[0] for x in self.cur.fetchall()]


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
            logger.warning(f'Parsed {section} in {end_time - start_time} seconds\n'
                           f'{end_stats[0] - start_stats[0]} new titles\n'
                           f'{end_stats[1] - start_stats[1]} new events')


if __name__ == '__main__':
    parser = YaParser(verbose=config['verbose'])
    for section in config['url_list']:
        parser.parse_news_section(section)

