import os
import sqlite3
import logging


logger = logging.getLogger('db')


class DB:
    def __init__(self, db_file, create_new=False):
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
            logger.error('Database connection error')
        return conn

    def create_tables(self):
        tables = {'events': ' (hash INT PRIMARY KEY, title TEXT, section TEXT, date DATE)',
                  'news_titles': ' (hash INT PRIMARY KEY, title TEXT, time TEXT, event INT)'}
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
        values = f'{title[0]}, \'{title[1]}\', \'{title[2]}\', \'{title[3]}\' '
        query = f'INSERT INTO news_titles(hash, title, time, event) ' \
                f'SELECT {values} ' \
                f'WHERE NOT EXISTS (SELECT 1 FROM news_titles WHERE hash={title[0]})'
        self.conn.execute(query)

    def update_database(self, section, data):
        self.conn.execute('BEGIN TRANSACTION;')
        for line in data:
            self.insert_event((line['event_hash'], line['event_title'], section, line['event_date']))
            self.insert_title((line['title_hash'], line['title'], line['time'], line['event_hash']))
        self.conn.execute('COMMIT;')
