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
        self.cur.execute('SELECT count(*) FROM news_titles UNION ALL SELECT count(*) FROM events ')
        return [x[0] for x in self.cur.fetchall()]


if __name__ == '__main__':
    db = DB('ya_parser/db3.db')
    print(db.select_stats())
