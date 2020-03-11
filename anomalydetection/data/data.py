
import pandas as pd


class Data:

    def __init__(self, path):
        self.path = path

    def read(self):

        return pd.read_csv(self.path, sep=';', header=0, skiprows=0)


