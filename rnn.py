import numpy as numpy
from rnn_utils import *

def rnn_cell_forward(xt, a_prev, param):
    '''
    Parameters:
    xt: data at time t (D, X)
    a_prev: hidden state at time t-1 (D, H)
    param: weights

    Returns:
    a_next: hidden state at time t (D, H)
    y_pred: prediction at time t (D, Y)
    cache: cache of parameters needed for backward pass
    '''

    W_hh = param["W_hh"] # (H, H)
    W_xh = param["W_xh"] # (X, H)
    W_hy = param["W_hy"] # (H, Y)
    b_h = param["b_h"]   # (H,)
    b_y = param["b_y"]   # (X,)

    h = np.matmul(xt, W_xh) + np.matmul(a_prev, W_hh) + b_h
    a_next = np.tanh(h)
    y_pred = softmax(np.matmul(a_next, W_hy) + b_y)
    cache = (a_next, a_prev, xt, param)
    
    return a_next, y_pred, cache

def rnn_cell_backward(da_next, cache):
    '''
    Parameters:
    da_next: gradient of loss w.r.t hidden state at time t
    cache: cache of parameters from forward pass

    Returns:
    d_x: grad of loss w.r.t input at time t
    d_a_prev: grad of loss w.r.t previous hidden state at time t - 1
    d_Whh: grad of loss w.r.t hidden-to-hidden weights
    d_Wxh: grad of loss w.r.t input-to-hidden weights
    d_bh: grad of loss w.r.t hidden bias

    '''

    (a_next, a_prev, xt, param) = cache
    W_hh = param["W_hh"] # (H, H)
    W_xh = param["W_xh"] # (X, H)
    W_hy = param["W_hy"] # (H, Y)
    b_h = param["b_h"]   # (H,)

    dh = da_next * (1 - np.square(a_next))

    d_x = np.matmul(dh, np.transpose(W_xh))
    d_Wxh = np.matmul(np.transpose(xt), dh)

    d_a_prev = np.matmul(dh, np.transpose(W_hh))
    d_Whh = np.matmul(np.transpose(a_prev), dh)

    d_bh = np.sum(dh, axis =0)

    return d_x, d_a_prev, d_Whh, d_Wxh, d_bh

def rnn_forward(x, a0, param):
    '''
    Parameters:
    x: data (D, X, T)
    a0: initial hidden state (D, H)
    param: dict of parameters

    Returns:
    a: output of forward pass (D, H, T)
    caches: list of caches from every time step

    '''
    (D, X, T) = x.shape
    (D, H) = a0.shape

    caches = {}
    a_next = np.zeros(a0.shape)
    a_next = a0.copy()

    a = np.zeros((D, H, T))

    for t in range(T):
        a_next, y_pred, cache = rnn_cell_forward(x[:, :, t], a_next, param)
        a[:, :, t] = a_next
        caches[t] = cache

    return a, caches

def rnn_backward(da, caches):
    '''
    Params:
    da: grad of loss w.r.t output
    caches: list of caches from forward pass

    Returns:
    dx, da0, dWhh, dWxh, dbh
    '''

    W_hh = param["W_hh"] # (H, H)
    W_xh = param["W_xh"] # (X, H)
    W_hy = param["W_hy"] # (H, Y)
    b_h = param["b_h"]   # (H,)
    
    (_, _, _, param) = caches[0]
    (D, H, T) = da.shape    
    (X, _) = W_xh.shape
    dx = np.zeros((D, X, T))
    dWhh = np.zeros(W_hh.shape)
    dWxh = np.zeros(W_xh.shape)
    dbh = np.zeros(b_h.shape)

    da0 = np.zeros((D, H))
    for t in range(T-1, -1, -1):
        d_x, da0, d_Whh, d_Wxh, d_bh = rnn_cell_backward(da0 + da[:, :, t] , caches[t])
        dWhh += d_Whh
        dWxh += d_Wxh
        dbh += d_bh
        dx[:, :, t] = d_x
    
    return dx, da0, dWhh, dWxh, dbh

def lstm_cell_forward(xt, a_prev, c_prev, param):
    pass

def lstm_cell_backward(da_next, dc_next, cache):
    pass

def lstm_forward(x, a0, param):
    pass

def lstm_backward(da, cache):
    pass

def test_gradient_rnn():
    np.random.seed(18)
    x = np.random.randn(10, 3, 7)
    a_prev = np.random.randn(10, 5)
    W_hh = np.random.randn(5,5)
    W_xh = np.random.randn(3,5)
    W_hy = np.random.randn(5,2)

    b_h = np.random.randn(5)
    b_y = np.random.randn(2)
    parameters = {"W_hh": W_hh, "W_xh": W_xh, "W_hy": W_hy, "b_h": b_h, "b_y": b_y}

    da = np.random.rand(10, 5, 7)

    a_next, caches = rnn_forward(x, a_prev, parameters)
    dx, da0, dWhh, dWxh, dbh = rnn_backward(da, caches)

    grad_a0 = eval_numerical_gradient_array(lambda a_prev: rnn_forward(x, a_prev, parameters)[0], a_prev, da)
    grad_x = eval_numerical_gradient_array(lambda x: rnn_forward(x, a_prev, parameters)[0], x, da)
    
    print (grad_a0 - da0)
    print (grad_x - dx)

def test_gradient_rnn_cell():
    np.random.seed(18)
    xt = np.random.randn(10, 3)
    a_prev = np.random.randn(10, 5)
    W_hh = np.random.randn(5,5)
    W_xh = np.random.randn(3,5)
    W_hy = np.random.randn(5,2)

    b_h = np.random.randn(5)
    b_y = np.random.randn(2)
    parameters = {"W_hh": W_hh, "W_xh": W_xh, "W_hy": W_hy, "b_h": b_h, "b_y": b_y}

    d_a_next = np.random.randn(10, 5)

    a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
    d_x, d_a_prev, d_Whh, d_Wxh, d_bh = rnn_cell_backward(d_a_next, cache)

    grad1, grad_3d = eval_grad(xt, a_prev, parameters)
    grad2 = eval_numerical_gradient_array(lambda xt: rnn_cell_forward(xt, a_prev, parameters)[0], xt, d_a_next, h = 0.0005)

    # # D x H, H x X
    # print (np.matmul(d_a_next, grad1))
    # D x H, D x H x X
    d_a_next_e = np.expand_dims(d_a_next, 1)
    gradient = np.squeeze(np.matmul(d_a_next_e, grad_3d))

    print (gradient - d_x)
    print (grad2 - d_x)

def eval_grad(xt, a_prev, parameters, h=0.00005):
    '''
    Parameters:
    xt: (D, X)

    Returns:
    grad_3d: grad of a_next w.r.t xt

    '''
    f = lambda xt: rnn_cell_forward(xt, a_prev, parameters)[0]

    (D, X) = xt.shape
    W_xh = parameters["W_xh"]
    (X, H) = W_xh.shape

    # (H, X)
    grad = np.zeros((H, X))

    # (D, H, X)
    grad_3d = np.zeros((D, H, X))

    for i in range(D):
        for j in range(X):
            oldval = xt[i, j]
            xt[i, j] = oldval + h
            pos = f(xt).copy()
            xt[i, j] = oldval - h
            neg = f(xt).copy()
            xt[i, j] = oldval
            # (D, H)
            dij = np.sum((pos - neg) / (2*h), axis=0)
            grad[:, j] += dij
            grad_3d[i, :, j] = dij

    return grad, grad_3d

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

if __name__ == "__main__":
    # test_gradient_rnn_cell()
    test_gradient_rnn()