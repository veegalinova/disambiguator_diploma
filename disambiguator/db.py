import sqlite3


class RutezDB:
    def __init__(self, db_file: str):
        self.cursor = sqlite3.connect(db_file).cursor()

    def select_words_db_ids(self, words):
        query_params = ','.join(['\'' + word.upper() + '\'' for word in words])
        query = """ SELECT entry_id, name, is_polysemic 
                    FROM text_entry 
                    WHERE name in ({}) """
        self.cursor.execute(query.format(query_params))
        rows = self.cursor.fetchall()
        result = {word.lower(): (idx, poly) for idx, word, poly in rows}
        return result

    def select_close_words(self, ids, max_relation_order=4):
        query_params = ','.join(ids)
        query = """ SELECT close_words.entry_id as base_id, 
                           entry_id_to as close_word_id, 
                           close_words.id_from as meaning_id, 
                           concepts.name as meaning_name
                    FROM close_words
                    INNER JOIN concepts on id_from = concepts.id
                    INNER JOIN text_entry text_entry_to on text_entry_to.entry_id = entry_id_to
                    WHERE close_words.relation_order <= {0} and base_id in ({1}) """
        self.cursor.execute(query.format(max_relation_order, query_params))
        rows = self.cursor.fetchall()

        close_words = {}
        close_words_to_meaning = {}
        meaning_id_to_word = {}

        for base_id, close_word_id, meaning_id, meaning_name in rows:
            if base_id not in close_words:
                close_words[base_id] = [close_word_id]
            else:
                close_words[base_id].append(close_word_id)

            if base_id not in close_words_to_meaning:
                close_words_to_meaning[base_id] = {close_word_id: meaning_id}
            else:
                close_words_to_meaning[base_id].update({close_word_id: meaning_id})

            if meaning_id not in meaning_id_to_word:
                meaning_id_to_word[meaning_id] = meaning_name

        return close_words, close_words_to_meaning, meaning_id_to_word

    def select_poly_entries_meanings(self, words=None, ids=None):
        if words:
            query_params = ','.join(['\'' + word.upper() + '\'' for word in words])
            query = """ SELECT entry_id
                        FROM text_entry 
                        WHERE name in ({}) """
            self.cursor.execute(query.format(query_params))
            ids = self.cursor.fetchall()
            ids = [str(id[0]) for id in ids]
        if not ids:
            return None
        query_params = ','.join(ids)
        query = """ SELECT DISTINCT close_words.entry_id as base_id, text_entry.name as base, 
                        concepts.name as meaning, close_words.id_from as meaning_id 
                    FROM close_words 
                    INNER JOIN concepts on id_from = concepts.id 
                    INNER JOIN text_entry on entry_id = text_entry.entry_id
                    WHERE entry_id in ({}) """
        self.cursor.execute(query.format(query_params))
        rows = self.cursor.fetchall()
        result = {}
        for base_id, base, meaning, meaning_id in rows:
            if base_id not in result:
                result[base_id] = [meaning]
            else:
                result[base_id].append(meaning)
        return result
