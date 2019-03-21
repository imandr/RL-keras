from generate import generator
from model import function_model, training_model

fm = function_model(2, 4)
tm = training_model(2, fm)
g = generator(200)

tm.fit_generator(g, steps_per_epoch = 100, epochs=100, verbose=1)