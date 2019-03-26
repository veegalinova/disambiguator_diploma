import re
import yaml
import logging
import sqlite3
from os.path import join
from collections import deque

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

with open("config.yml", 'r') as config_file:
    config = yaml.load(config_file)

FROM_PATTERN = re.compile(r'from="(\d*)"')
TO_PATTERN = re.compile(r'to="(\d*)"')
NAME_PATTERN = re.compile(r'name="(\w*)"')
ASP_PATTERN = re.compile(r'asp=\"([\w ]*)\"')
ID_PATTERN = re.compile(r'concept_id="(\d*)"')
ENTRY_ID_PATTERN = re.compile(r'entry_id="(\d*)"')
TEXT_ENTRY_PATTERN = re.compile(
    r'<entry id=\"(\d*)\">\s*'
    r'<name>(.*)</name>\s*'
    r'<lemma>(.*)</lemma>\s*'
    r'<main_word>(.*)</main_word>\s*'
    r'<synt_type>(\w*)</synt_type>\s*'
    r'<pos_string>(.*)</pos_string>\s*'
)
CONCEPTS_PATTERN = re.compile(
    r'<concept id="(\d*)">\s*'
    r'<name>(.*)</name>\s*'
    r'<gloss>(.*)</gloss>\s*'
    r'<domain>(.*)</domain>'
)


class DBCreator:
    def __init__(self):
        with sqlite3.connect(config['database']) as self.conn:
            self.cursor = self.conn.cursor()
            logger.info('Created db')
            self.create_tables()
            logger.info('Created tables')
            self.load_concepts()
            logger.info('Filled concepts')
            self.load_synonyms()
            logger.info('Filled synonyms')
            self.load_entries()
            logger.info('Filled entries')
            self.load_relations()
            logger.info('Filled relations')
            self.load_close_words()
            logger.info('Filled close words')

    def create_tables(self):
        self.cursor.execute("""
        CREATE TABLE concepts(
            id INTEGER primary key,
            name TEXT not null,
            gloss TEXT,
            domain TEXT
        );
        """)
        self.cursor.execute("""
        CREATE INDEX concept_names_idx ON concepts(name);
        """)
        self.cursor.execute("""
        CREATE TABLE text_entry(
            entry_id INTEGER primary key,
            name TEXT,
            lemma TEXT,
            main_word TEXT,
            synt_type TEXT,
            pos_string TEXT,
            is_polysemic integer
        );
        """)
        self.cursor.execute("""
        CREATE TABLE synonyms(
            concept_id INTEGER references concepts,
            entry_id INTEGER references text_entry (entry_id)
        );
        """)
        self.cursor.execute("""
        CREATE TABLE relations(
            id_from INTEGER references concepts,
            id_to INTEGER references concepts,
            name TEXT,
            asp TEXT
        );
        """)

    @staticmethod
    def _file_window(file, num_lines):
        window = deque((file.readline() for _ in range(1, num_lines)), maxlen=num_lines)
        text = "".join(window)
        return text

    def load_concepts(self):
        with open(join(config['rutez_dir'], 'concepts.xml'), 'r', encoding="utf-8") as concepts:
            self.cursor.execute('begin')
            for _ in concepts:
                line = self._file_window(concepts, 5)
                search_line = CONCEPTS_PATTERN.search(line)
                if search_line is None:
                    continue
                self.cursor.execute("""
                INSERT INTO concepts(id, name, gloss, domain) VALUES (?, ?, ?, ?)
                """, [search_line.group(1), search_line.group(2), search_line.group(3), search_line.group(4)])
            self.cursor.execute('commit')

    def load_relations(self):
        with open(join(config['rutez_dir'], 'relations.xml'), 'r', encoding="utf-8") as relations:
            self.cursor.execute('begin')
            for line in relations:
                from_search = FROM_PATTERN.search(line) or None
                if from_search is None:
                    continue
                from_search = from_search.group(1)
                to_search = TO_PATTERN.search(line).group(1)
                name_search = NAME_PATTERN.search(line).group(1)
                asp_search = ASP_PATTERN.search(line).group(1)
                if asp_search == ' ':
                    asp_search = None
                self.cursor.execute("""
                    INSERT INTO relations(id_from, id_to, name, asp) 
                    VALUES (?, ?, ?, ?)""", [from_search, to_search, name_search, asp_search]
                                    )
            self.cursor.execute('commit')

    def load_synonyms(self):
        with open(join(config['rutez_dir'], 'synonyms.xml'), 'r', encoding="utf-8") as synonyms:
            self.cursor.execute('begin')
            for line in synonyms:
                id_search = ID_PATTERN.search(line) or None
                entry_id_search = ENTRY_ID_PATTERN.search(line) or None
                if id_search is None or entry_id_search is None:
                    continue
                id_search = id_search.group(1)
                entry_id_search = entry_id_search.group(1)
                self.cursor.execute("""
                    INSERT INTO synonyms(concept_id, entry_id)
                    VALUES (?, ?)""", [id_search, entry_id_search]
                                    )
            self.cursor.execute('commit')

    def _delete_duplicates(self):
        self.cursor.execute("""
                DELETE from close_words
                WHERE rowid not in(
                    SELECT MIN(ROWID)
                    FROM close_words
                    GROUP BY id_from, entry_id, id_to
                );
                """)

    def _add_new_relation(self, order=2):
        self.cursor.execute("""
        INSERT INTO close_words
            SELECT DISTINCT close_words.id_from, close_words.entry_id,
                relations.id_to, s1.entry_id, {0}
            FROM relations
              INNER JOIN close_words ON relations.id_from = close_words.id_to and close_words.relation_order = {1}
              INNER JOIN synonyms s1 on relations.id_to = s1.concept_id
              INNER JOIN text_entry t1 on s1.entry_id = t1.entry_id and t1.is_polysemic = 0
            WHERE close_words.id_from != relations.id_to
        """.format(order, order-1))

    def load_entries(self):
        self.cursor.execute('begin')
        with open(join(config['rutez_dir'], 'text_entry.xml'), 'r', encoding="utf-8") as text_entry:
            for _ in text_entry:
                line = self._file_window(text_entry, 7)
                search_line = TEXT_ENTRY_PATTERN.search(line)
                if search_line is None:
                    continue
                self.cursor.execute("""
                    INSERT INTO text_entry 
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                    [
                                        search_line.group(1),
                                        search_line.group(2),
                                        search_line.group(3),
                                        search_line.group(4),
                                        search_line.group(5),
                                        search_line.group(6),
                                        0
                                    ]
                                    )
        self.cursor.execute('commit')
        self.cursor.execute('begin')
        self.cursor.execute("""
                UPDATE text_entry SET is_polysemic = 1
                WHERE text_entry.entry_id in(
                    SELECT entry_id as entry
                    FROM synonyms
                    GROUP BY entry
                    HAVING COUNT(*) > 1
                )
                """)
        self.cursor.execute('commit')

    def load_close_words(self):
        self.cursor.execute("""
        CREATE TABLE close_words(
          id_from integer, 
          entry_id integer,
          id_to integer, 
          entry_id_to integer, 
          relation_order integer
        );""")

        self.cursor.execute('begin')

        self.cursor.execute("""
         INSERT INTO close_words
            SELECT relations.id_from, s1.entry_id, relations.id_to, s2.entry_id, 1
            FROM relations
              INNER JOIN synonyms s1 ON relations.id_from = s1.concept_id
              INNER JOIN synonyms s2 on relations.id_to = s2.concept_id
              INNER JOIN text_entry t1 on s1.entry_id = t1.entry_id and t1.is_polysemic = 1
              INNER JOIN text_entry t2 on s2.entry_id = t2.entry_id and t2.is_polysemic = 0;
        """)
        logger.info('1st relation')

        self._add_new_relation(order=2)
        self._delete_duplicates()
        logger.info('2nd relation')

        self._add_new_relation(order=3)
        self._delete_duplicates()
        logger.info('3rd relation')

        self._add_new_relation(order=4)
        self._delete_duplicates()
        logger.info('4th relation')

        self.cursor.execute('commit')


def create_database():
    db = DBCreator()


if __name__ == '__main__':
    create_database()
