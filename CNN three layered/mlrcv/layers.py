from builtins import range
import numpy as np

def affine_forward(x, w, b):
    out = None
    
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)

    out = np.dot(x_reshaped, w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    
    x, w, b = cache
    dx, dw, db = None, None, None
   
    # pass
    N = x.shape[0]
    # print("N: ", x.shape)
    x_reshaped = x.reshape(N, -1)
    # print("x_reshaped: ", x_reshaped.shape)

    dx = (np.dot(dout, w.T)).reshape(x.shape)
    # print("dx: ", dx.shape)
    dw = (np.dot(x_reshaped.T, dout))
    db = np.sum(dout, axis=0)

   
    return dx, dw, db


def relu_forward(x):
    out = None
    

    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
  
    dx, x = None, cache
    dx =dout*(x>0).astype(float)
    return dx

def conv_forward_naive(x, w, b, conv_param):
    
    out = None
   
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    Hout = 1 + (H + 2 * pad - HH) // stride
    Wout = 1 + (W + 2 * pad - WW) // stride

    x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    out = np.zeros((N, F, Hout, Wout))
    
    for n in range(N):
        for f in range(F):
            for i in range(Hout):
                for j in range(Wout):
                    x_slice = x_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]
    
    return out, (x, w, b, conv_param)

    


def conv_backward_naive(dout, cache):
  
    dx, dw, db = None, None, None
   
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    Hout = 1 + (H + 2 * pad - HH) // stride
    Wout = 1 + (W + 2 * pad - WW) // stride

    x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(Hout):
                for j in range(Wout):
                    x_slice = x_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                    dx_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[f] * dout[n, f, i, j]
                    dw[f] += x_slice * dout[n, f, i, j]
                    db[f] += dout[n, f, i, j]

    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]


    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    
    out = None
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    Hout = 1 + (H - pool_height) // stride
    Wout = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, Hout, Wout))

    for n in range(N):
        for c in range(C):
            for i in range(Hout):
                for j in range(Wout):
                    w_start = i * stride
                    h_start = j * stride
                    w_end = w_start + pool_height
                    h_end = h_start + pool_width

                    x_slice = x[n, c, w_start:w_end, h_start:h_end]
                    out[n, c, i, j] = np.max(x_slice)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
   
    dx = None
    
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    Hout = 1 + (H - pool_height) // stride
    Wout = 1 + (W - pool_width) //stride

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(Hout):
                for j in range(Wout):
                    w_start = i * stride
                    h_start = j * stride
                    
                    x_slice = x[n, c, w_start:w_start+pool_height, h_start:h_start+pool_width]
                    mask = (x_slice == np.max(x_slice))
                    dx[n, c, w_start:w_start+pool_height, h_start:h_start+pool_width] += mask * dout[n, c, i, j]


    return dx

def svm_loss(x, y):
    
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
