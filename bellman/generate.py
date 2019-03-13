import numpy as np

delta = 0.1

def generator(mbsize):
    while True:
        s0 = np.random.random((mbsize,2))
        
        x0 = s0[:,0]
        t0 = s0[:,1]
        
        direction = np.random.randint(0, 2, mbsize)
        
        s1 = s0.copy()
        s1[direction==0, 0] += delta            # x+
        s1[direction==1, 1] += delta            # y+

        x1 = s1[:,0]
        y1 = s1[:,1]
        
        r = np.zeros((mbsize,))
        
        for i, si in enumerate(s1):
            xi, yi = si
            if xi > 1.0:
                r[i] = 1.0 if yi < 0.2 else -1.0
            elif yi > 1.0:
                r[i] = 1.0 if xi < 0.1 else -1.0
                
        final = np.zeros((mbsize,))
        final[r != 0] = 1.0

        
        yield [s0, s1, final], r
        
        
def sample(nx, nt, model):
    x = np.arange(0.0, 1.0, 1.0/nx)
    t = np.arange(0.0, 1.0, 1.0/nt)
    xg, tg = np.meshgrid(x, t)
    s = np.array((xg.reshape((-1,)),tg.reshape((-1,)))).T
    #print "s:",s.shape,s
    z = np.max(model.predict_on_batch(s), axis=-1)
    return xg, tg, z[:,0].reshape(xg.shape)
    