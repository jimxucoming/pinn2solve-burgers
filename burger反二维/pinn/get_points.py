import numpy as np
from scipy.io import loadmat

def domain(n,pars):

    t = np.random.uniform(low=0,high=pars['tf'],size=(n,1))
    x = np.random.uniform(low=pars['xi'],high=pars['xf'],size=(n,1))
    y = np.random.uniform(low=pars['yi'],high=pars['yf'],size=(n,1))

    X = np.hstack((t,x,y))

    return X

def solution(n,pars):

    data = loadmat(pars['solution_file'])

    [x, t, y] = np.meshgrid(data['x'],data['t'],data['y'])

    t = t.flatten().reshape(-1, 1)
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    u = data['uref'].transpose((2,1,0)).flatten().reshape(-1, 1)
    v = data['vref'].transpose((2,1,0)).flatten().reshape(-1, 1)

    ind = np.random.choice(t.shape[0], size=n)

    X = np.hstack((t[ind],x[ind],y[ind]))
    Y = np.hstack((u[ind],v[ind]))

    return X, Y


def solution_plotx(t_in,x_in,pars):

    data = loadmat(pars['solution_file'])

    [x, t, y] = np.meshgrid(data['x'],data['t'],data['y'])

    t = t.flatten().reshape(-1, 1)
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    u = data['uref'].transpose((2,1,0)).flatten().reshape(-1, 1)
    v = data['vref'].transpose((2,1,0)).flatten().reshape(-1, 1)

    inx = np.arange(x_in*100*101+101*101*t_in*100,x_in*100*101+101*101*t_in*100+101).astype('int64')

    X = np.hstack((t[inx],x[inx],y[inx]))
    Y = np.hstack((u[inx],v[inx]))

    return X, Y


def solution_ploty(t_in,y_in,pars):

    data = loadmat(pars['solution_file'])

    [x, t, y] = np.meshgrid(data['x'],data['t'],data['y'])

    t = t.flatten().reshape(-1, 1)
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    u = data['uref'].transpose((2,1,0)).flatten().reshape(-1, 1)
    v = data['vref'].transpose((2,1,0)).flatten().reshape(-1, 1)

    iny = np.arange(y_in*100+101*101*t_in*100,y_in*100+101*101*t_in*100+101*101,101).astype('int64')

    X = np.hstack((t[iny],x[iny],y[iny]))
    Y = np.hstack((u[iny],v[iny]))

    return X, Y
