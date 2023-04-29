from __future__ import absolute_import, print_function

import tvm
import numpy as np
import tvm.topi as topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name='A')
    B = tvm.te.placeholder(shape, dtype=dtype, name='B')
    C = tvm.te.compute(A.shape, lambda *i: A(*i) * B(*i))
    
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name='A')
    B = tvm.te.const(const_k)
    C = tvm.te.compute(A.shape, lambda *i: A(*i) + B)
    
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name='A')
    B = tvm.te.const(const_k)
    C = tvm.te.compute(A.shape, lambda *i: A(*i) * B)
    
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """Hint: use tvm.max, tvm.te.const(0, A.dtype)"""
    A = tvm.te.placeholder(shape, dtype=dtype, name='A')
    B = tvm.te.const(0)
    C = tvm.te.compute(A.shape, lambda *i: tvm.te.max(A(*i), B))
    
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """Hint: use tvm.select"""
    A = tvm.te.placeholder(shape, dtype=dtype, name='A')
    B = tvm.te.placeholder(shape, dtype=dtype, name='B')
    C = tvm.te.const(0)
    D = tvm.te.compute(A.shape, lambda *i: tvm.te.if_then_else(A(*i) > C, B(*i), C))
    
    s = tvm.te.create_schedule(D.op)
    f = tvm.build(s, [A, B, D], tgt, target_host=tgt_host, name=func_name)
    return f
    

def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    A = tvm.te.placeholder((shapeA[0], shapeA[1]), dtype=dtype, name='A')
    B = tvm.te.placeholder((shapeB[0], shapeB[1]), dtype=dtype, name='B')
    
    if transposeA == False and transposeB == False:
        k = tvm.te.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.te.compute((shapeA[0], shapeB[1]), lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=k))
    elif transposeA == True and transposeB == False:
        k = tvm.te.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.te.compute((shapeA[1], shapeB[1]), lambda i, j: tvm.te.sum(A[k, i] * B[k, j], axis=k))
    elif transposeA == False and transposeB == True:
        k = tvm.te.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.te.compute((shapeA[0], shapeB[0]), lambda i, j: tvm.te.sum(A[i, k] * B[j, k], axis=k))
    else:
        k = tvm.te.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.te.compute((shapeA[1], shapeB[0]), lambda i, j: tvm.te.sum(A[k, i] * B[j, k], axis=k))

    s = tvm.te.create_schedule(C.op)
    
    ###########
    # code for parallel scheduling
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=4, y_factor=4)
    xk, yk = s[C].split(k, factor=4)
    s[C].reorder(xo, yo, xk, xi, yk, yi)
    s[C].parallel(xo)
    s[C].vectorize(xi)
    s[C].unroll(yk)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    ###########
    
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

#######################
#     Convolutions    #
#######################
def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF
    HO = H-R+1
    WO = W-S+1

    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    input_mat = tvm.te.placeholder(shapeX, dtype=dtype, name='input_mat')
    conv_kernel = tvm.te.placeholder(shapeF, dtype=dtype, name='conv_kernel')
    
    dr = tvm.te.reduce_axis((0, R), name='dr')
    ds = tvm.te.reduce_axis((0, S), name='ds')
    dc = tvm.te.reduce_axis((0, C), name='dc')
    output_mat = tvm.te.compute((N, M, HO, WO), 
                             lambda n, m, i, j: tvm.te.sum(input_mat[n, dc, i+dr, j+ds] * conv_kernel[m, dc, dr, ds], axis=[dr, ds, dc]))
    
    # drs = tvm.te.reduce_axis((0, R*S), name='drs')
    # dc = tvm.te.reduce_axis((0, C), name='dc')
    # output_mat = tvm.te.compute((N, M, HO, WO), 
    #                         lambda n, m, i, j: tvm.te.sum(input_mat[n,dc,i+tvm.te.floordiv(drs,S),j+drs%S] * conv_kernel[m,dc,tvm.te.floordiv(drs,S),drs%S], axis=[drs,dc]))

    s = tvm.te.create_schedule(output_mat.op)
    f = tvm.build(s, [input_mat, conv_kernel, output_mat], tgt, target_host=tgt_host, name=func_name)
    
    return f
    
def make_conv2d_grad_x(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert (shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF
    HO = H-R+1
    WO = W-S+1
    
    """This function is currently still BUGGY so don't use it!!!"""
    input_mat = tvm.te.placeholder(shapeX, dtype=dtype, name='input_mat')
    conv_kernel = tvm.te.placeholder(shapeF, dtype=dtype, name='conv_kernel')
    output_grad_mat = tvm.te.placeholder((N, M, HO, WO), dtype=dtype, name='output_grad_mat')

    dm = tvm.te.reduce_axis((0, M), name='dm')

    input_mat_grad = tvm.te.compute(shapeX, lambda n, c, h, w:
                                    tvm.te.sum(output_grad_mat[n, dm, h - R + 1 + dm, w - S + 1 + dm] * conv_kernel[dm, c, R - 1 - dm, S - 1 - dm], axis=dm))

    s = tvm.te.create_schedule(input_mat_grad.op)
    f = tvm.build(s, [output_grad_mat, input_mat, conv_kernel, input_mat_grad], tgt, target_host=tgt_host, name=func_name)

    return f

def make_conv2d_grad_f(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert (shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF
    HO = H-R+1
    WO = W-S+1

    input_mat = tvm.te.placeholder(shapeX, dtype=dtype, name='input_mat')
    conv_kernel = tvm.te.placeholder(shapeF, dtype=dtype, name='conv_kernel')
    output_grad_mat = tvm.te.placeholder((N, M, HO, WO), dtype=dtype, name='output_grad_mat')

    dn = tvm.te.reduce_axis((0, N), name='dn')
    di = tvm.te.reduce_axis((0, HO), name='di')
    dj = tvm.te.reduce_axis((0, WO), name='dj')
    conv_kernel_grad = tvm.te.compute(shapeF, 
        lambda m, c, r, s: tvm.te.sum(input_mat[dn, c, di+r, dj+s]*output_grad_mat[dn,m,di,dj], axis=[dn,di,dj]))

    s = tvm.te.create_schedule(conv_kernel_grad.op)
    f = tvm.build(s, [input_mat, conv_kernel, output_grad_mat, conv_kernel_grad], tgt, target_host=tgt_host, name=func_name)

    return f

def make_maxpool2d(shapeX, pool_size, stride, tgt, tgt_host, func_name, dtype="float32"):
    N, C, H, W = shapeX
    
    input_mat = tvm.te.placeholder(shapeX, dtype=dtype, name='input_mat')
    
    di = tvm.te.reduce_axis((0, pool_size), name='di')
    dj = tvm.te.reduce_axis((0, pool_size), name='dj')
    oh = H // stride
    ow = W // stride
    output_mat = tvm.te.compute((N, C, oh, ow),
                                lambda n, c, h, w: tvm.te.max(
                                    input_mat[n, c, h*stride+di, w*stride+dj],
                                    axis=[di, dj]
                                ))
    
    s = tvm.te.create_schedule(output_mat.op)
    f = tvm.build(s, [input_mat, output_mat], tgt, target_host=tgt_host, name=func_name)
    
    return f

def make_maxpool2d_grad(shapeX, pool_size, stride, tgt, tgt_host, func_name, dtype="float32"):
    N, C, H, W = shapeX
    
    input_mat = tvm.te.placeholder(shapeX, dtype=dtype, name='input_mat')
    output_mat = tvm.te.placeholder((N, C, H//stride, W//stride), dtype=dtype, name='output_mat')
    output_grad_mat = tvm.te.placeholder((N, C, H//stride, W//stride), dtype=dtype, name='output_grad_mat')
    
    """this implementation is still buggy so don't use it for now"""
    
    di = tvm.te.reduce_axis((0, pool_size), name='di')
    dj = tvm.te.reduce_axis((0, pool_size), name='dj')
    oh = H // stride
    ow = W // stride
    
    max_pool_patch = tvm.te.compute((N, C, oh, ow, pool_size, pool_size),
                                    lambda n, c, h, w, i, j: input_mat[n, c, h*stride+i, w*stride+j])
    
    max_pool_patch_reshape = tvm.te.reshape(max_pool_patch, (N, C, oh, ow, -1))
    
    max_index = tvm.te.compute((N, C, oh, ow, 2),
                               lambda n, c, h, w: tvm.te.cast(tvm.te.stack(tvm.te.max(max_pool_patch_reshape[n, c, h, w], axis=-1),
                                                                           tvm.te.argmax(max_pool_patch_reshape[n, c, h, w], axis=-1)),
                                                             'int32'))
    
    input_mat_grad = tvm.te.compute(shapeX,
                                    lambda n, c, h, w: tvm.te.sum(
                                        tvm.te.if_then_else(
                                            max_index[n, c, h, w, 0] == di*pool_size + dj,
                                            output_grad_mat[n, c, h, w],
                                            0),
                                        axis=[di, dj]
                                    ))
    
    s = tvm.te.create_schedule(input_mat_grad.op)
    f = tvm.build(s, [input_mat, output_mat, output_grad_mat, input_mat_grad], tgt, target_host=tgt_host, name=func_name)
    
    return f


def make_batchnorm2d(shapeX, tgt, tgt_host, func_name, dtype="float32"):
    N, C, H, W = shapeX

    input_mat = tvm.te.placeholder(shapeX, dtype=dtype, name='input_mat')
    gamma = tvm.te.placeholder((1, C, 1, 1), dtype=dtype, name='gamma')
    beta = tvm.te.placeholder((1, C, 1, 1), dtype=dtype, name='beta')
    eps = tvm.tir.const(1e-5, dtype=dtype)
    
    # calculate mean
    dn1 = tvm.te.reduce_axis((0, N), name='dn1')
    dh1 = tvm.te.reduce_axis((0, H), name='dh1')
    dw1 = tvm.te.reduce_axis((0, W), name='dw1')

    sum_ = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: tvm.te.sum(input_mat[dn1, c, dh1, dw1], axis=[dn1, dh1, dw1]))
    mean = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: sum_[0, c, 0, 0] / (N*H*W))

    # calculate var
    dn2 = tvm.te.reduce_axis((0, N), name='dn2')
    dh2 = tvm.te.reduce_axis((0, H), name='dh2')
    dw2 = tvm.te.reduce_axis((0, W), name='dw2')

    # var = E[X^2] - E[X]^2 = E[(X-mu)^2]
    square = tvm.te.compute((N, C, H, W), lambda n, c, h, w: tvm.te.power(input_mat[n, c, h, w] - mean[0, c, 0, 0], 2))
    square_sum = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: tvm.te.sum(square[dn2, c, dh2, dw2], axis=[dn2, dh2, dw2]))
    var = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: square_sum[0, c, 0, 0] / (N*H*W))

    # Normalize and scale
    upper = input_mat - mean
    lower = topi.sqrt(var + eps)
    x_norm = upper / lower
    output_mat = gamma * x_norm + beta
    
    s = tvm.te.create_schedule(output_mat.op)
    f = tvm.build(s, [input_mat, gamma, beta, output_mat], tgt, target_host=tgt_host, name=func_name)

    return f

def make_batchnorm2d_grad(shapeX, tgt, tgt_host, func_name, dtype="float32"):
    """this function is still buggy so don't use it for now"""
    N, C, H, W = shapeX
    
    # arr_x, arr_gamma, arr_beta, arr_dout, arr_dx, arr_dgamma, arr_dbeta
    input_mat = tvm.te.placeholder(shapeX, dtype=dtype, name='input_mat')
    gamma = tvm.te.placeholder((1, C, 1, 1), dtype=dtype, name='gamma')
    mean = tvm.te.placeholder((1, C, 1, 1), dtype=dtype, name='mean')
    var = tvm.te.placeholder((1, C, 1, 1), dtype=dtype, name='var')
    dout = tvm.te.placeholder(shapeX, dtype=dtype, name='dout')
    eps = tvm.tir.const(1e-5, dtype=dtype)
    
    # std = np.sqrt(var + eps)
    std = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: tvm.te.sqrt(var[0, c, 0, 0] + eps))
    # istd = 1.0 / std
    istd = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: 1.0 / std[0, c, 0, 0])
    
    # x_norm = (x - mean) / std
    x_norm = tvm.te.compute(shapeX, lambda n, c, h, w: (input_mat[n, c, h, w] - mean[0, c, 0, 0]) * istd[0, c, 0, 0])
    
    # dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
    dn1 = tvm.te.reduce_axis((0, N), name='dn1')
    dh1 = tvm.te.reduce_axis((0, H), name='dh1')
    dw1 = tvm.te.reduce_axis((0, W), name='dw1')
    dgamma = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: tvm.te.sum(dout[dn1, c, dh1, dw1] * x_norm[dn1, c, dh1, dw1], axis=[dn1, dh1, dw1]))
    
    # dx_norm = dout * gamma
    dx_norm = tvm.te.compute(shapeX, lambda n, c, h, w: dout[n, c, h, w] * gamma[0, c, 0, 0])
    
    # dbeta = np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1)
    dn2 = tvm.te.reduce_axis((0, N), name='dn2')
    dh2 = tvm.te.reduce_axis((0, H), name='dh2')
    dw2 = tvm.te.reduce_axis((0, W), name='dw2')
    dbeta = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: tvm.te.sum(dout[dn2, c, dh2, dw2], axis=[dn2, dh2, dw2]))
    
    # istdsq = (-0.5) * istd**3
    istdsq = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: (-0.5) * tvm.te.power(istd[0, c, 0, 0], 3))
    # mul = dx_norm * (x - mean) * istdsq
    mul = tvm.te.compute(shapeX, lambda n, c, h, w: dx_norm[n, c, h, w] * (input_mat[n, c, h, w] - mean[0, c, 0, 0]) * istdsq[0, c, 0, 0])
    
    # dvar = np.sum(mul, axis=(0, 2, 3), keepdims=True)
    dn3 = tvm.te.reduce_axis((0, N), name='dn3')
    dh3 = tvm.te.reduce_axis((0, H), name='dh3')
    dw3 = tvm.te.reduce_axis((0, W), name='dw3')
    dvar = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: tvm.te.sum(mul[dn3, c, dh3, dw3], axis=[dn3, dh3, dw3]))
    
    # left = np.sum(dx_norm * (-istd), axis=(0, 2, 3), keepdims=True)
    left = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: tvm.te.sum(dx_norm[dn3, c, dh3, dw3] * (-istd[0, c, 0, 0]), axis=[dn3, dh3, dw3]))
    
    # inner = -2.0 * (x - mean)
    inner = tvm.te.compute(shapeX, lambda n, c, h, w: -2.0 * (input_mat[n, c, h, w] - mean[0, c, 0, 0]))
    # part = np.mean(infer, axis=(0, 2, 3), keepdims=True)
    dn4 = tvm.te.reduce_axis((0, N), name='dn4')
    dh4 = tvm.te.reduce_axis((0, H), name='dh4')
    dw4 = tvm.te.reduce_axis((0, W), name='dw4')
    part = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: tvm.te.sum(inner[dn4, c, dh4, dw4], axis=[dn4, dh4, dw4]))
    # right = dvar * part
    right = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: dvar[0, c, 0, 0] * part[0, c, 0, 0])
    
    # dmean = left + right
    dmean = tvm.te.compute((1, C, 1, 1), lambda n, c, h, w: left[0, c, 0, 0] + right[0, c, 0, 0])
    # dx = dx_norm * istd + dvar * 2.0 * (x - mean) / (N * H * W) + dmean / (N * H * W)
    dx = tvm.te.compute(shapeX, lambda n, c, h, w: dx_norm[n, c, h, w] * istd[0, c, 0, 0] + dvar[0, c, 0, 0] * 2.0 * (input_mat[n, c, h, w] - mean[0, c, 0, 0]) / (N*H*W) + dmean[0, c, 0, 0] / (N*H*W))
    
    s = tvm.te.create_schedule(dx.op)
    f = tvm.build(s, [input_mat, gamma, mean, var, dout, dx, dgamma, dbeta], tgt, target_host=tgt_host, name=func_name)
    return f

def make_flatten(shape, tgt, tgt_host, func_name, dtype="float32"):
    N, C, H, W = shape
    
    x = tvm.te.placeholder(shape, dtype=dtype, name="x")
    out = tvm.te.compute((N, C*H*W), lambda n, c: x[n, c // (H*W), (c // W) % H, c % W], name="out")
    
    s = tvm.te.create_schedule(out.op)
    f = tvm.build(s, [x, out], tgt, target_host=tgt_host, name=func_name)
    return f

def make_flatten_grad(shape, tgt, tgt_host, func_name, dtype="float32"):
    N, C, H, W = shape
    
    dout = tvm.te.placeholder((N, C*H*W), dtype=dtype, name="dout")
    x = tvm.te.placeholder(shape, dtype=dtype, name="x")
    dx = tvm.te.compute(shape, lambda n, c, h, w: dout[n, c * H * W + h * W + w], name="dx")
    
    s = tvm.te.create_schedule(dx.op)
    f = tvm.build(s, [x, dout, dx], tgt, target_host=tgt_host, name=func_name)
    return f

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    
    # e_x = np.exp(x - np.max(x))
    k1 = tvm.te.reduce_axis((0, shape[1]), name='k1')
    max_elem = tvm.te.compute((shape[0],), lambda i: tvm.te.max(A[i, k1], axis=k1))
    exp_elem = tvm.te.compute(shape, lambda i, j: tvm.te.exp(A[i, j] - max_elem[i]))
    
    # softmax(x)= e_x / e_x.sum()
    k2 = tvm.te.reduce_axis((0, shape[1]), name='k2')
    sum_exp = tvm.te.compute((shape[0],), lambda i: tvm.te.sum(exp_elem[i, k2], axis=k2))
    B = tvm.te.compute(shape, lambda i, j: exp_elem[i, j] / sum_exp[i])
    
    s = tvm.te.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f
    

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """Hint: output shape should be (1,)"""
    
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    # e_x = np.exp(x - np.max(x))
    k1 = tvm.te.reduce_axis((0, shape[1]), name='k1')
    max_elem_a = tvm.te.compute((shape[0],), lambda i: tvm.te.max(A[i, k1], axis=k1))
    exp_elem_a = tvm.te.compute(shape, lambda i, j: tvm.te.exp(A[i, j] - max_elem_a[i]))
    # softmax(x)= e_x / e_x.sum()
    k2 = tvm.te.reduce_axis((0, shape[1]), name='k2')
    sum_exp_a = tvm.te.compute((shape[0],), lambda i: tvm.te.sum(exp_elem_a[i, k2], axis=k2))
    softmax_a = tvm.te.compute(shape, lambda i, j: exp_elem_a[i, j] / sum_exp_a[i])
    # np.log(autodiff.softmax_func(y)), axis=1)
    log_softmax_a = tvm.te.compute(shape, lambda i, j: tvm.te.log(softmax_a[i, j]))
    
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    # np.sum(ans, keepdims=True)
    k3 = tvm.te.reduce_axis((0, shape[1]), name='k3')
    log_softmax_a_sum = tvm.te.compute((shape[0],), lambda i: tvm.te.sum(B[i, k3] * log_softmax_a[i, k3], axis=k3))
    # np.mean(-ans, axis=1) -> sum(ans / len(ans), axis=1)
    k4 = tvm.te.reduce_axis((0, shape[0]), name='k4')
    softmax_cross_entropy = tvm.te.compute((1,), lambda i: tvm.te.sum(-log_softmax_a_sum[k4] / shape[0], axis=k4))
    
    s = tvm.te.create_schedule(softmax_cross_entropy.op)
    f = tvm.build(s, [A, B, softmax_cross_entropy], tgt, target_host=tgt_host, name=func_name)
    return f
    
def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.te.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.te.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.te.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.te.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f

