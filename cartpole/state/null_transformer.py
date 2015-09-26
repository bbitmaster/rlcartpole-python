#this class implements a tabular storage for a Qsa table
import numpy as np
import math

class null_transformer(object):
    def __init__(self):
        self.transform_class = None

    def set_transform_class(self,transform_class):
        self.transform_class = transform_class

    def transform(self,state):
        if(self.transform_class is None):
            return state
        else:
            return self.transform_class.transform(state)

if __name__ == '__main__':
    pass
