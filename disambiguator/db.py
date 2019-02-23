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
        query = 'SELECT id_from, concepts.name, entry_id, entry_id_to, concepts_2.name ' \
                'FROM relations_from_meanings ' \
                'INNER JOIN concepts on id_from = concepts.id ' \
                'INNER JOIN concepts concepts_2 on id_to = concepts_2.id ' \
                'WHERE entry_id in ({})'
        self.cursor.execute(query.format(query_params))
        rows = self.cursor.fetchall()
        close_words = {}
        idx_to_meaning = {}
        midx_to_meaning = {}
        idx_to_word = {}

        for concept_id_from, meaning, meaning_entry_id, entry_id_to, name_to in rows:
            if meaning_entry_id not in close_words:
                close_words[meaning_entry_id] = [entry_id_to]
            else:
                close_words[meaning_entry_id].append(entry_id_to)

            if entry_id_to not in idx_to_word:
                idx_to_word[entry_id_to] = name_to

            if entry_id_to not in idx_to_meaning:
                idx_to_meaning[entry_id_to] = meaning_entry_id

            if meaning_entry_id not in midx_to_meaning:
                midx_to_meaning[meaning_entry_id] = meaning

        return close_words, idx_to_meaning, midx_to_meaning, idx_to_word


    def select_poly_entries_meanings(self, words):
        query_params = ','.join(['\'' + word.upper() + '\'' for word in words])
        query = 'SELECT entry ' \
                'FROM text_entry ' \
                'WHERE name in ({}) '
        self.cursor.execute(query.format(query_params))
        ids = self.cursor.fetchall()
        if not ids:
            return None
        ids = [str(id[0]) for id in ids]
        query_params = ','.join(ids)
        query = 'SELECT id_from, concepts.name, text_entry.name ' \
                'FROM relations_from_meanings ' \
                'INNER JOIN concepts on id_from = concepts.id ' \
                'INNER JOIN text_entry on entry_id = text_entry.entry ' \
                'WHERE entry_id in ({}) ' \
                'ORDER BY entry_id '
        self.cursor.execute(query.format(query_params))
        rows = self.cursor.fetchall()
        result = {}
        for idx, meaning, entry in rows:
            result.update({idx: entry.replace(',', '*') + ': ' + meaning.replace(',', '*')})
        return result
