import sqlite3


class RutezDB:
    def __init__(self, db_file: str):
        self.cursor = sqlite3.connect(db_file).cursor()

    def select_words_db_ids(self, words):
        query_params = ','.join(['\'' + word.upper() + '\'' for word in words])
        query = 'SELECT entry, name, is_polysemic ' \
                'FROM text_entry ' \
                'WHERE name in ({})'
        self.cursor.execute(query.format(query_params))
        rows = self.cursor.fetchall()
        result = {word.lower(): (idx, poly) for idx, word, poly in rows}
        return result

    def select_close_words(self, ids):
        query_params = ','.join(ids)
        query = 'SELECT id_from, concepts.name, entry_id, entry_id_to ' \
                'FROM relations_from_meanings ' \
                'INNER JOIN concepts on id_from = concepts.id ' \
                'WHERE entry_id in ({})'
        self.cursor.execute(query.format(query_params))
        rows = self.cursor.fetchall()
        result = {}
        meanings = {}
        for idx, meaning, entry_id, close_word in rows:
            if entry_id not in result:
                result[entry_id] = {close_word: idx}
            else:
                result[entry_id][close_word] = idx
            if idx not in meanings:
                meanings[idx] = meaning
        return result, meanings

    def query_word_meanings(self, ids):
        query_params = ','.join(ids)
        query = 'SELECT id_from, concepts.name, entry_id ' \
                'FROM relations_from_meanings ' \
                'INNER JOIN concepts on id_from = concepts.id ' \
                'WHERE entry_id in ({})'
        self.cursor.execute(query.format(query_params))
        rows = self.cursor.fetchall()
        result = {}
        meanings = {}
        for idx, meaning, entry_id in rows:
            if entry_id not in result:
                result[entry_id] = {idx: meaning}
            else:
                result[entry_id].update({idx: meaning})

        return result
