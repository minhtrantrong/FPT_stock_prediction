from matplotlib import scale
from numpy.core.numerictypes import sctype2char
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class PG_DB():
    def __init__(self, pg_config, table):
        self.config = pg_config
        self.tb = table
        self.df = self.__get_table()
        self.values, self.scaler, self.scaled = self.__MaxMinScale()
        
        
    def __get_table(self):
        query = "SELECT * FROM {}".format(self.tb)
        pg_conn = psycopg2.connect(self.config)
        df = pd.read_sql(query, con=pg_conn, parse_dates = True)
        df = df.set_index("date")[::-1]
        return df

    def __MaxMinScale(self):
        values = self.df.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        return values, scaler, scaled

    def review(self):
        print(self.df.head())
        print(self.df.describe())
        return

    def plot(self):
        fig, axes = plt.subplots(nrows=2, ncols=1)
        self.df["open_price"].plot(figsize=(20,8), ax=axes[0])
        self.df["ex_value"].plot(figsize=(20,8), ax=axes[1])
        return  

    def dataset_gen(self, num_steps_in=1, num_steps_out=1):
        '''
        This function reformats the dataset the way it can be fed to the LSTM.
        '''
        col_names = [name for name in self.df.columns]
    
        df = pd.DataFrame(self.scaled)
        columns, names = list(), list()
        
        # input sequence (t-n, ... t-1)
        for i in range(num_steps_in, 0, -1):

            columns.append(df.shift(i))
            names += ['%s(t-%d)' % (n, i) for n in col_names]
        
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, num_steps_out):
            columns.append(df.shift(-i))
            if i == 0:
                names += [('%s(t)' % n) for n in col_names]
            else:
                names += [('%s(t+%d)' % (n, i)) for n in col_names]
    
        dataset = pd.concat(columns, axis=1)
        dataset.columns = names
        
        return dataset
