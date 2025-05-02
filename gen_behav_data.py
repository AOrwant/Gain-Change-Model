import pickle
import numpy as np
from model import *

with open("trained_pop.pkl", 'rb') as input:
    trained_pop = pickle.load(input)

static_pop = Population()


t = "inversion"
trials = 10000

stimRF_num = []
stimOpp_num = []
correct = []
selected = []
resp = []
rule = []

for trial in range(trials):
    context = np.random.choice([0, 1])
    shape0 = np.random.choice(SHAPES)
    color0 = np.random.choice(COLORS)

    if trial % 1000 == 0:
         print(100 * trial/trials, '%')

    if t == "feature":
        if context == 0:
            shape1 = np.random.choice([s for s in SHAPES if s != shape0])
            color1 = color0
        if context== 1:
            shape1 = shape0
            color1 = np.random.choice([c for c in COLORS if c != color0])

    if t == "inversion":
        shape1 = np.random.choice(SHAPES)
        color1 = np.random.choice(COLORS)

    correct_choice, _, _  = get_correct(shape0, color0, shape1, color1, context, t) # 0 if stimRF, 1 if stimOpp

    # Model choice
    RF_resp = trained_pop.resp(context, color0, shape0)
    Opp_resp = static_pop.resp(context, color1, shape1)

    if RF_resp < Opp_resp:
            choice = 1
    else:
            choice = 0

    #Adjust shape and color to the directions used for animal behavioral data
    sh0 = abs(4 - shape0)
    sh1 = abs(4 - shape1)

    # Find stimRF_num and stimOpp_num
    RF = (5 * sh0) + color0 + 1 
    Opp = (5 * sh1) + color1 + 1

    stimRF_num.append(RF)
    stimOpp_num.append(Opp)
    correct.append(correct_choice + 1)
    selected.append(choice + 1)

    # Find neural responses

    resps = []

    for col in range(trained_pop.col_cats): 
            for shp in range(trained_pop.shp_cats):
                resps.append(trained_pop.tiling[col][shp].resp(color0, shape0, context))

    resp.append(resps)

    # rule 

    rule.append(context)


resp = np.asarray(resp)
np.savetxt("resp.csv", resp, delimiter = ",")

params = np.asarray([stimRF_num, stimOpp_num, correct, selected, rule])
np.savetxt("params.csv", params, delimiter = ",")

