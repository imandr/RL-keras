import numpy as np
from keras.models import model_from_config


def max_valid(q, valid):
    return q[valid[np.argmax(q[valid])]]

def best_valid_action(q, valid):
    return valid[np.argmax(q[valid])]

def format_batch(batch):
    
    # if the model takes multiple inputs, then batch is a list of lists like [x1, x2, x3].
    # We need to convert that into [batch(x1), batch(x2, batch(x3)]
    
    if len(batch) == 0:
        return batch
        
    row = batch[0]
    if isinstance(row, (list, tuple)):
        return [np.array(x) for x in zip(*batch)]
    else:
        return np.array(batch)

def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone
