from scipy import stats
import numpy as np


COLORS = [0, 1, 2, 3, 4] #GRAY (0) to BLUE (4)
SHAPES = [0, 1, 2, 3, 4] #TRIANGLE (0) to CIRCLE (4)
LIMIT = 0.15 #fraction gain change may change
UNIT_CHANGE = 0.0001 #scaling for augmentation

class Neuron: 
    '''
    A Neuron is an object which has tuning functions for shape and color with gain changes that depend on context.
    Neurons give a response to both color and shape according to their tuning curves, with their total response being a
    sum of the two. 

    These may be augmented once created *only* to alter gain changes. 
    '''

    def __init__(self, color_selectivity, shape_selectivity, color_gain = 1, shape_gain = 1, sd = 0.5):
        self.col_sel = color_selectivity
        self.shape_sel = shape_selectivity
        self.col_gains = [color_gain, color_gain]
        self.shape_gains = [shape_gain, shape_gain]
        self.sd = sd

    def col_resp(self, color_value, context):
        gain = self.col_gains[context]
        return gain * stats.norm.pdf(color_value, self.col_sel, self.sd)
    
    def shape_resp(self, shape_value, context):
        gain = self.shape_gains[context]
        return gain * stats.norm.pdf(shape_value, self.shape_sel, self.sd)
    
    def resp(self, col_val, shape_val, context):
        return self.shape_resp(shape_val, context) + self.col_resp(col_val, context)
    
    def augment(self, feature, size, context):
        if feature == "color":
            if 1 - LIMIT <= self.col_gains[context] + size <= 1 + LIMIT:
                self.col_gains[context] += size
        if feature == "shape":
            if 1 - LIMIT <= self.shape_gains[context] + size <= 1 + LIMIT:
                self.shape_gains[context] += size
        # for the neurons tuned most to this feature, augments their gain up or down as needed based on performance

class Population: 
    '''
    Grid of neurons that tiles the space
    '''
    def __init__(self, color_cats = 20, shape_cats = 20, buffer = 0.5):
        #Initializes the grid of neurons which tile the space in specified units (w specified buffer)

        col_range = np.linspace(COLORS[0] - buffer, COLORS[4] + buffer, color_cats)
        shape_range = np.linspace(SHAPES[0] - buffer, SHAPES[4] + buffer, shape_cats)

        grid = []
        for i, col in enumerate(col_range):
            grid.append([])
            for j, shp in enumerate(shape_range):
                grid[i].append(Neuron(col, shp))

        self.tiling = grid
        self.col_cats = color_cats
        self.shp_cats = shape_cats
        self.window = round(color_cats/5)

    def resp(self, context, col_val, shape_val):
        sum = 0
        for col in range(self.col_cats):
            for shp in range(self.shp_cats):
                neuron = self.tiling[col][shp]
                sum += neuron.resp(col_val, shape_val, context)

        return sum

    def augment(self, correct, choice, col_val, shp_val, col_diff, shp_diff, context):
        sign = -1
        if correct and choice == 0: 
            sign = 1
        if not correct and choice == 1:
            sign = 1
        
        col_change = sign * UNIT_CHANGE
        shp_change = sign * UNIT_CHANGE

        w = self.window
        col_st = col_val * w 
        shp_st = shp_val * w

        for color_row in range(self.col_cats):
            for shape_col in range(self.shp_cats):
                if col_st <= color_row <= col_st + w:
                    self.tiling[color_row][shape_col].augment("color", col_change, context)
                if shp_st <= shape_col <= shp_st + 2:
                    self.tiling[color_row][shape_col].augment("shape", shp_change, context)

def run_trial(updating_pop, static_pop, sh0, col0, sh1, col1, context, t):
    '''
    Given both populations and the sh/col for both stimulus inputs, as well as context and "t," which tracks whether 
    contexts indicate feature relevance or inversion, will compare the responses of the two populations and update the 
    non-static population accordingly. 

    UPDATES population in response to trial correctness
    RETURNS whether or not the correct option was chosen
    '''
    correct_choice, col_diff, shape_diff = get_correct(sh0, col0, sh1, col1, context, t)
    U = updating_pop.resp(context, col0, sh0)
    S = static_pop.resp(context, col1, sh1)
    if S > U:
        choice = 1 
    else:
        choice = 0
    
    correct = (choice == correct_choice)

    updating_pop.augment(correct, choice, col0, sh0, col_diff, shape_diff, context)
    return correct

def get_correct(sh0, col0, sh1, col1, context, t):
    '''
    given a context, returns which stimulus (0 or 1) is correct. 
    "t" clarifies whether we are operating under inversions as context or relevant feature as context
    '''
    col_diff = col1 - col0
    shape_diff = sh1 - sh0

    comparison = col_diff + shape_diff
    if comparison > 0:
        correct = 1
    elif comparison <0:
        correct = 0
    else:
        correct = np.random.choice([0, 1])

    if t == "inversion" and context == 1:
        correct = abs(correct - 1)
    
    return correct, abs(col_diff), abs(shape_diff)

def train(updating_pop, static_pop, num_trials, t = "feature"):
    '''
    Creates stimuli and, according to the type of context (feature-based or inversion-based), assigns stimuli then 
    checks model performance and updates accordingly 

    Does this for the number of trials specified

    FOR FEATURE AS CONTEXT: 
        context 0 indicates a shape trial
        context 1 indicates a color trial
    
    FOR INVERSION AS CONTEXT: 
        context 0 indicates classic rule (blue circle is best)
        context 1 indicates inverted rule (gray triangle is best)
    '''
    counter = 0
    corrects = 0
    for _ in range(num_trials):
        context = np.random.choice([0, 1])
        shape0 = np.random.choice(SHAPES)
        color0 = np.random.choice(COLORS)

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

        counter += 1
        c = run_trial(updating_pop, static_pop, shape0, color0, shape1, color1, context, t)
        if c:
            corrects += 1
        if counter >= 50:
            print(corrects/counter)
            corrects = 0
            counter = 0