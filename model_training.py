import pickle
from model import *

updating_pop = Population()
static_pop = Population()

train(updating_pop, static_pop, 40000, "inversion")

with open("trained_pop.pkl", 'wb') as output: 
    trained_pop = updating_pop
    pickle.dump(trained_pop, output, pickle.HIGHEST_PROTOCOL)