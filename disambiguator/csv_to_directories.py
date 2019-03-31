import pandas as pd
import os

from disambiguator.experiment_runner import config


def res_to_directories(csv_file, path):
    df = pd.read_csv(csv_file)
    i = 0
    for idx, row in df.iterrows():
        res_path = path + row['normalized'] + '/' + row['meaning']
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        with open('{0}/{1}.csv'.format(res_path, i), 'w') as file:
            file.write(str(row['position']) + ',  ' + str(row['word']) + ', ' + str(row['text']))
            file.write('\n')
            i += 1


if __name__ == '__main__':
    res_to_directories(config['output'], 'data/directories/')
