import pickle
from model import *

with open("trained_pop.pkl", 'rb') as input:
    trained_pop = pickle.load(input)

print(trained_pop.tiling[2][2].col_gains)

