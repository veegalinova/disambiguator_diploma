import sqlite3
import re
import yaml
from os.path import join
from collections import deque

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
    def __init__(self, db_file):
        with sqlite3.connect(db_file) as self.conn:
            self.cursor = self.conn.cursor()
            self.create_tables()
            self.load_concepts()
            self.load_entries()
            self.load_synonyms()
            self.load_relations()

    def create_tables(self):
        self.cursor.execute("""
        CREATE TABLE concepts(
            id INTEGER primary key,
            name TEXT not null,
            gloss TEXT,
            domain TEXT
        );
        
        CREATE INDEX concept_names_idx ON concepts(name);
        
        CREATE TABLE text_entry(
            entry_id INTEGER primary key,
            name TEXT,
            lemma TEXT,
            main_word TEXT,
            synt_type TEXT,
            pos_string TEXT,
            is_polysemic integer DEFAULT 0
        );
        
        CREATE TABLE synonyms(
            concept_id INTEGER references concepts,
            entry_id INTEGER references text_entry (entry_id)
        );
        
        CREATE TABLE relations(
            id_from INTEGER references concepts,
            id_to INTEGER references concepts,
            name TEXT,
            asp TEXT
        );
        
        UPDATE text_entry SET is_polysemic = 1
        WHERE text_entry.entry in(
            SELECT entry_id as entry
            FROM synonyms
            GROUP BY entry
            HAVING COUNT(*) > 1
        );
        """)

    @staticmethod
    def _file_window(file, num_lines):
        window = deque((file.readline() for _ in range(1, num_lines)), maxlen=num_lines)
        text = "".join(window)
        return text

    def load_concepts(self):
        with open(join(config['rutez_xml'], 'concepts.xml'), 'r') as concepts:
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
        with open(join(config['rutez_xml'], 'relations.xml'), 'r') as relations:
            self.cursor.execute('begin')
            for line in relations:
                from_search = FROM_PATTERN.search(line).group(1) or None
                if from_search is None:
                    continue
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
        with open(join(config['rutez_xml'], 'synonyms.xml'), 'r') as synonyms:
            self.cursor.execute('begin')
            for line in synonyms:
                id_search = ID_PATTERN.search(line).group(1) or None
                entry_id_search = ENTRY_ID_PATTERN.search(line).group(1) or None
                if id_search is None or entry_id_search is None:
                    continue
                self.cursor.execute("""
                    INSERT INTO synonyms(concept_id, entry_id)
                    VALUES (?, ?)""", [id_search, entry_id_search]
                                    )
            self.cursor.execute('commit')

    def load_entries(self):
        self.cursor.execute('begin')
        with open(join(config['rutez_xml'], 'text_entry.xml'), 'r') as text_entry:
            for _ in text_entry:
                line = self._file_window(text_entry, 7)
                search_line = TEXT_ENTRY_PATTERN.search(line)
                if search_line is None:
                    continue
                self.cursor.execute("""
                    INSERT INTO text_entry 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                                    [
                                        search_line.group(1),
                                        search_line.group(2),
                                        search_line.group(3),
                                        search_line.group(4),
                                        search_line.group(5),
                                        search_line.group(6)
                                    ]
                                    )
        self.cursor.execute('commit')

