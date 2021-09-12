from sqlalchemy import create_engine
import pandas as pd
from contextlib import contextmanager
import numpy as np


class ExperimentDB(object):

    def __init__(self, db_path):
        self._db_path = db_path
    
    @contextmanager
    def db_connection(self):

        db = create_engine(f'sqlite:///{self._db_path}')
        conn = db.connect()
        try:
            yield conn
        except Exception as e:
            print('\033[91mFailed to access DB file\033[0m')
        finally:
            conn.close()

    @property
    def db_path(self):
        return self._db_path

    def save_execution_info(self, start_time, parameters):

        with self.db_connection() as conn:
            experiment_row = {'start_time': start_time}
            experiment_row.update(parameters)

            if 'ancestor_genotype' in parameters:
                experiment_row['ancestor_genotype'] = ''.join([str(i) for i in experiment_row['ancestor_genotype']])
                
            if 'classification_image_dimensions' in parameters:
                experiment_row['classification_image_dimensions'] = str(parameters['classification_image_dimensions'])

            try:
                existing_rows = pd.read_sql_table('execution_info', con=conn)
            except ValueError:
                existing_rows = pd.DataFrame()

            existing_rows = existing_rows.append(experiment_row, ignore_index=True)

            existing_rows.to_sql('execution_info', con=conn, index=False, if_exists='replace')

    def save_seranns_info(self, seranns_info):
    
        with self.db_connection() as conn:
            sequence_str = seranns_info['genotype'].map(lambda x: str(x.tolist()))
            seranns_info.assign(genotype=sequence_str).to_sql('serann', con=conn, if_exists='append')

    def save_generation_info(self, generation_info):
    
        with self.db_connection() as conn:
            pd.Series(generation_info).to_frame().T.to_sql('generations', index=False, con=conn, if_exists='append')

    def save_model_metrics(self, model_name, metrics):
        with self.db_connection() as conn:
            metrics.assign(model=model_name).to_sql('model_metrics', con=conn, index=False, if_exists='append')

    def get_last_execution_info(self):
        with self.db_connection() as conn:

            int_columns = ['genotype_size', 'initial_offspring_pool_size', 'max_serann_parameters',
                           'max_serann_tokens', 'num_classification_classes', 'num_generations', 'num_seranns',
                           'offspring_pool_size_factor', 'random_seed', 'training_batch_size', 'training_epochs']

            float_columns = ['error_correction_probability', 'selection_pressure']

            try:
                raw = pd.read_sql(f'select * from execution_info order by start_time desc limit 1',
                                  con=conn).iloc[0].copy()

                for column in raw.keys():
                    if column in int_columns:
                        raw[column] = raw[column].astype(int)
                    elif column in float_columns:
                        raw[column] = raw[column].astype(float)

                if 'classification_image_dimensions' not in raw and 'classification_image_height' in raw:

                    raw['classification_image_dimensions'] = list(raw[['classification_image_height',
                                                                  'classification_image_width']].astype(int))
                else:
                    raw['classification_image_dimensions'] = list(np.int8(eval(raw['classification_image_dimensions'])))

                if 'ancestor_genotype' in raw:
                    raw['ancestor_genotype'] = [int(s) for s in raw['ancestor_genotype']]

                if 'initial_offspring_pool_size' not in raw:
                    raw['initial_offspring_pool_size'] = 10

                if 'offspring_pool_size_factor' not in raw or raw['offspring_pool_size_factor'] == 1:
                    raw['offspring_pool_size_factor'] = 3

                return raw

            except:
                return pd.DataFrame()

    def get_generations_count(self):
        with self.db_connection() as conn:
            try:
                return pd.read_sql('select count(*) from generations', con=conn).iloc[0, 0]
            except:
                return 0

    def get_executions_count(self):
        with self.db_connection() as conn:
            try:
                return pd.read_sql('select count(*) from execution_info', con=conn).iloc[0, 0]
            except:
                return 0

    def get_serann_by_generation(self, generation):

        with self.db_connection() as conn:
            df = pd.read_sql(f'select * from serann where generation = {generation}', con=conn).set_index('id')
    
        df['genotype'] = df['genotype'].map(lambda x: np.array(eval(f'{x.replace("nan", "np.NaN")}')).astype('float16'))
        df['is_valid'] = df['is_valid'].astype(bool)
        return df
