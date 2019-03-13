import numpy as np

"""
t: 0...1
x: -1...1


dz/dt = x
dz/dx = (1-t)*x
z|t>1 = 1.0
"""


def generator(mbsize):
    delta = 0.2
    while True:
        s0 = np.random.random((mbsize,2))
        s0[:,0] = s0[:,0]*2-1
        
        x0 = s0[:,0]
        t0 = s0[:,1]
        
        direction = np.random.randint(0, 4, mbsize)
        
        s1 = s0.copy()
        s1[direction==0, 0] += delta            # x+
        s1[direction==1, 0] -= delta            # x-
        s1[direction==2, 1] += delta            # t+
        s1[direction==3, 1] -= delta            # t-

        x1 = s1[:,0]
        t1 = s1[:,1]

        dz = np.empty_like(x0)
        dz[direction==0] = ((1-t0)*x0*delta)[direction==0]
        dz[direction==1] = (-(1-t0)*x0*delta)[direction==1]
        dz[direction==2] = (x0*delta)[direction==2]
        dz[direction==3] = (-x0*delta)[direction==3]
        
        boundary = (t1 > 1.0)       # * (np.abs(x1) < 0.1)
        dz[boundary] = 1.0
        
        yield [s0, s1, np.asarray(boundary, dtype=np.float32)], dz
        
        
def sample(nx, nt, model):
    x = np.arange(-1.0, 1.0, 2.0/nx)
    t = np.arange(-0.1, 1.1, 1.2/nt)
    xg, tg = np.meshgrid(x, t)
    s = np.array((xg.reshape((-1,)),tg.reshape((-1,)))).T
    print "s:",s.shape,s
    z = model.predict_on_batch(s)
    return xg, tg, z[:,0].reshape(xg.shape)
    