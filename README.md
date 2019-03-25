# disambiguator

## Structure:

1. **disambiguator/create_db.py** - Creates sqlite database from tezaurus xml files, specified in config.yaml
2. **disambiguator/db.py** - Tezaurus database queries
3. **disambiguator/runner.py** - Determines experiments and scorers
4. **disambiguator/text_processor.py** - Processes text, finds polysemous words in text, loops through its' close words, suggests meaning, scores it 
5. **ya_parser/database.py** - Creates sqlite database to store parsing results
6. **ya_parser/parser.py** - Parses sections of news.yandex.ru and adds new news titles to database
