import numpy as np

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

