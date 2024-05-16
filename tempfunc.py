import numpy as np

def tempfunc_one(time):
    return 100*np.exp(-144 * ((600-time)/3600)**2 )

def tempfunc_two(time):
    if time <= 600:
        return 88.42
    else: 
        return 0
    
