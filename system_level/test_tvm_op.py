import sys
import numpy as np
import tvm
from . import autodiff, tvm_op
from time import time

tgt_host="llvm"
tgt="llvm"
dtype = "float32"
ctx = tvm.device(tgt, 0)


# def test_matrix_elementwise_add():
#     shape = (500, 200)
#     x = np.random.uniform(0, 10, size=shape).astype(dtype)
#     y = np.random.uniform(0, 10, size=shape).astype(dtype)
#     z = np.zeros(shape).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)
#     arr_z = tvm.nd.array(z)
#     elemwise_add = tvm_op.make_elemwise_add(shape, tgt, tgt_host, "elem_add")
#     elemwise_add(arr_x, arr_y, arr_z)
#     z = arr_z.asnumpy()
#     np.testing.assert_allclose(x + y, z, rtol=1e-5)


# def test_matrix_elementwise_add_by_const():
#     shape = (2000, 3000)
#     x = np.random.uniform(0, 10, size=shape).astype(dtype)
#     const_val = np.random.uniform(0, 10)
#     y = np.zeros(shape).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)
#     elemwise_add_by_const = tvm_op.make_elemwise_add_by_const(shape, const_val, tgt, tgt_host, "elem_add_by_const")
#     elemwise_add_by_const(arr_x, arr_y)
#     y = arr_y.asnumpy()
#     np.testing.assert_allclose(x + const_val, y, rtol=1e-5)


# def test_matrix_elementwise_mul():
#     shape = (500, 200)
#     x = np.random.uniform(0, 10, size=shape).astype(dtype)
#     y = np.random.uniform(0, 10, size=shape).astype(dtype)
#     z = np.zeros(shape).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)
#     arr_z = tvm.nd.array(z)
#     elemwise_mul = tvm_op.make_elemwise_mul(shape, tgt, tgt_host, "elem_add")
#     elemwise_mul(arr_x, arr_y, arr_z)
#     z = arr_z.asnumpy()
#     np.testing.assert_allclose(x * y, z, rtol=1e-5)


# def test_matrix_elementwise_mul_by_const():
#     shape = (2000, 3000)
#     x = np.random.uniform(0, 10, size=shape).astype(dtype)
#     const_val = np.random.uniform(0, 10)
#     y = np.zeros(shape).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)
#     elemwise_mul_by_const = tvm_op.make_elemwise_mul_by_const(shape, const_val, tgt, tgt_host, "elem_mul_by_const")
#     elemwise_mul_by_const(arr_x, arr_y)
#     y = arr_y.asnumpy()
#     np.testing.assert_allclose(x * const_val, y, rtol=1e-5)


# def test_relu():
#     shape = (2000, 2500)
#     x = np.random.uniform(-1, 1, shape).astype(dtype)
#     y = np.zeros(shape).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)
#     relu = tvm_op.make_relu(shape, tgt, tgt_host, "relu")
#     relu(arr_x, arr_y)
#     y = arr_y.asnumpy()
#     np.testing.assert_allclose(np.maximum(x, 0).astype(dtype), y)


# def test_relu_gradient():
#     shape = (2000, 2500)
#     x = np.random.uniform(-1, 1, shape).astype(dtype)
#     grad_x = np.random.uniform(-5, 5, shape).astype(dtype)
#     y = np.zeros(shape).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_grad_x = tvm.nd.array(grad_x)
#     arr_y = tvm.nd.array(y)
#     relu_gradient = tvm_op.make_relu_gradient(shape, tgt, tgt_host, "relu_gradient")
#     relu_gradient(arr_x, arr_grad_x, arr_y)
#     y = arr_y.asnumpy()
#     np.testing.assert_allclose(((x > 0) * grad_x).astype(dtype), y)

# def test_matrix_multiply():
    
#     list1 = []
#     list2 = []
#     list3 = []
#     list4 = []
#     total_list = []
    
#     for iter in range(10):
    
#         start_time = time()
        
#         shapeX = (500, 700)
#         shapeY = (700, 1000)
#         shapeZ = (500, 1000)
#         x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#         y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
#         z = np.zeros(shapeZ).astype(dtype)
#         arr_x = tvm.nd.array(x)
#         arr_y = tvm.nd.array(y)
#         arr_z = tvm.nd.array(z)
#         matrix_mul = tvm_op.make_matrix_mul(shapeX, False, shapeY, False, tgt, tgt_host, "matrix_mul")
#         matrix_mul(arr_x, arr_y, arr_z)
#         z = arr_z.asnumpy()
#         np.testing.assert_allclose(np.dot(x, y), z, rtol=1e-5)
        
#         checkpoint1 = time()

#         shapeX = (1000, 500)
#         shapeY = (2000, 500)
#         shapeZ = (1000, 2000)
#         x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#         y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
#         z = np.zeros(shapeZ).astype(dtype)
#         arr_x = tvm.nd.array(x)
#         arr_y = tvm.nd.array(y)
#         arr_z = tvm.nd.array(z)
#         matrix_mul = tvm_op.make_matrix_mul(shapeX, False, shapeY, True, tgt, tgt_host, "matrix_mul")
#         matrix_mul(arr_x, arr_y, arr_z)
#         z = arr_z.asnumpy()
#         np.testing.assert_allclose(np.dot(x, np.transpose(y)), z, rtol=1e-5)
        
#         checkpoint2 = time()
        
#         shapeX = (500, 1000)
#         shapeY = (500, 2000)
#         shapeZ = (1000, 2000)   
#         x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#         y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
#         z = np.zeros(shapeZ).astype(dtype)
#         arr_x = tvm.nd.array(x)
#         arr_y = tvm.nd.array(y)
#         arr_z = tvm.nd.array(z)
#         matrix_mul = tvm_op.make_matrix_mul(shapeX, True, shapeY, False, tgt, tgt_host, "matrix_mul")
#         matrix_mul(arr_x, arr_y, arr_z)
#         z = arr_z.asnumpy()
#         np.testing.assert_allclose(np.dot(np.transpose(x), y), z, rtol=1e-5)
        
#         checkpoint3 = time()
        
#         shapeX = (500, 1000)
#         shapeY = (2000, 500)
#         shapeZ = (1000, 2000)
#         x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#         y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
#         z = np.zeros(shapeZ).astype(dtype)
#         arr_x = tvm.nd.array(x)
#         arr_y = tvm.nd.array(y)
#         arr_z = tvm.nd.array(z)
#         matrix_mul = tvm_op.make_matrix_mul(shapeX, True, shapeY, True, tgt, tgt_host, "matrix_mul")
#         matrix_mul(arr_x, arr_y, arr_z)
#         z = arr_z.asnumpy()
#         np.testing.assert_allclose(np.dot(np.transpose(x), np.transpose(y)), z, rtol=1e-5)
        
#         end_time = time()
        
#         list1.append(checkpoint1 - start_time)
#         avg_time1 = sum(list1) / len(list1)
#         list2.append(checkpoint2 - checkpoint1)
#         avg_time2 = sum(list2) / len(list2)
#         list3.append(checkpoint3 - checkpoint2)
#         avg_time3 = sum(list3) / len(list3)
#         list4.append(end_time - checkpoint3)
#         avg_time4 = sum(list4) / len(list4)
#         total_list.append(end_time - start_time)
#         avg_time = sum(total_list) / len(total_list)

#     with open("tvm_result.txt", "w") as f:
#         f.write("1000*500 * 700*1000 average: " + str(round(avg_time1, 3)) + "\n")
#         f.write("1000*500 * 2000*500 average: " + str(round(avg_time2, 3)) + "\n")
#         f.write("500*1000 * 500*2000 average: " + str(round(avg_time3, 3)) + "\n")
#         f.write("500*1000 * 2000*500 average: " + str(round(avg_time4, 3)) + "\n")
#         f.write("Total time: " + str(round(avg_time, 3)) + "\n")

# def np_maxpool2d(x, pool_size=2, stride=2):
#     N, C, H, W = x.shape
#     HO = int((H - pool_size) / stride + 1)
#     WO = int((W - pool_size) / stride + 1)
#     out = np.zeros((N, C, HO, WO))

#     for n in range(N):
#         for c in range(C):
#             for i in range(HO):
#                 for j in range(WO):
#                     start_i = i * stride
#                     start_j = j * stride
#                     patch = x[n, c, start_i:start_i+pool_size, start_j:start_j+pool_size]
#                     out[n, c, i, j] = np.max(patch)

#     return out

# def test_maxpool2d():
#     shapeX = (30, 3, 14, 14)
#     pool_size = 2
#     stride = 2
#     shapeY = (30, 3, 7, 7)
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     y = np.zeros(shapeY).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)
   
#     maxpool2d = tvm_op.make_maxpool2d(shapeX, pool_size, stride, tgt, tgt_host, "maxpool2d")
#     maxpool2d(arr_x, arr_y)
#     y = arr_y.asnumpy()
#     np.testing.assert_allclose(np_maxpool2d(x, pool_size, stride), y, rtol=1e-5)
    
# def np_maxpool2d_backward(grad_output, x, pool_size=2, stride=2):
#     N, C, H, W = x.shape
#     _, _, HO, WO = grad_output.shape
#     grad_input = np.zeros_like(x)

#     for n in range(N):
#         for c in range(C):
#             for i in range(HO):
#                 for j in range(WO):
#                     start_i = i * stride
#                     start_j = j * stride
#                     patch = x[n, c, start_i:start_i+pool_size, start_j:start_j+pool_size]
#                     max_index = np.unravel_index(np.argmax(patch), patch.shape)
#                     grad_input[n, c, start_i+max_index[0], start_j+max_index[1]] += grad_output[n, c, i, j]

#     return grad_input

# def test_maxpool2d_backward():
#     shapeX = (30, 3, 14, 14)
#     pool_size = 2
#     stride = 2
#     shapeY = (30, 3, 7, 7)
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     grad_output = np.random.uniform(0, 10, size=shapeY).astype(dtype)
#     grad_input = np.zeros(shapeX).astype(dtype)
#     arr_x = tvm.nd.array(x)

#     arr_grad_output = tvm.nd.array(grad_output)
#     arr_grad_input = tvm.nd.array(grad_input)
    
#     maxpool2d_backward = tvm_op.make_maxpool2d_grad(shapeX, pool_size, stride, tgt, tgt_host, "maxpool2d_backward")
#     maxpool2d_backward(arr_x, arr_grad_output, arr_grad_input)
#     grad_input = arr_grad_input.asnumpy()
#     np.testing.assert_allclose(np_maxpool2d_backward(grad_output, x, pool_size, stride), grad_input, rtol=1e-5)

# def batch_norm_2d(x, gamma, beta, eps):
#     N, C, H, W = x.shape

#     # Compute mean and variance
#     mu = np.mean(x, axis=(0, 2, 3)).reshape(1, C, 1, 1)
#     var = np.var(x, axis=(0, 2, 3)).reshape(1, C, 1, 1)

#     # Normalize and scale
#     x_norm = (x - mu) / np.sqrt(var + eps)
#     out = gamma.reshape(1, C, 1, 1) * x_norm + beta.reshape(1, C, 1, 1)

#     return out

# def test_batch_norm_2d():
#     shapeX = (2, 3, 4, 4)
#     shapeGamma = (1, 3, 1, 1)
#     shapeBeta = (1, 3, 1, 1)
    
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     gamma = np.random.uniform(0, 10, size=shapeGamma).astype(dtype)
#     beta = np.random.uniform(0, 10, size=shapeBeta).astype(dtype)
#     mean = np.zeros((1, 3, 1, 1)).astype(dtype)
#     var = np.zeros((1, 3, 1, 1)).astype(dtype)
#     out = np.zeros(shapeX).astype(dtype)
    
#     arr_x = tvm.nd.array(x)
#     arr_gamma = tvm.nd.array(gamma)
#     arr_beta = tvm.nd.array(beta)
#     arr_mean = tvm.nd.array(mean)
#     arr_var = tvm.nd.array(var)
#     arr_out = tvm.nd.array(out)
    
#     tvm_batch_norm_2d = tvm_op.make_batchnorm2d(shapeX, tgt, tgt_host, "batch_norm_2d")
#     tvm_batch_norm_2d(arr_x, arr_gamma, arr_beta, arr_mean, arr_var, arr_out)
    
#     out = arr_out.asnumpy()
#     np.testing.assert_allclose(batch_norm_2d(x, gamma, beta, 1e-5), out, rtol=1e-4)

# def batch_norm_2d_backward(dout, x, gamma, mean, var, eps=1e-5):
#     # Compute the batch size
#     N, C, H, W = x.shape
    
#     # Compute the standard deviation and inverse of the standard deviation
#     std = np.sqrt(var + eps)
#     istd = 1.0 / std
    
#     # Compute the normalized input
#     x_norm = (x - mean) / std
    
#     # Compute the gradients with respect to gamma and beta
#     dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
#     dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    
#     # Compute the gradient with respect to the input
#     dx_norm = dout * gamma
#     dvar = np.sum(dx_norm * (x - mean) * (-0.5) * istd**3, axis=(0, 2, 3), keepdims=True)
#     dmean = np.sum(dx_norm * (-istd), axis=(0, 2, 3), keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=(0, 2, 3), keepdims=True)
#     dx = dx_norm * istd + dvar * 2.0 * (x - mean) / (N * H * W) + dmean / (N * H * W)
    
#     return dx_norm


# def test_batch_norm_2d_backward():
#     shapeX = (2, 3, 4, 4)
#     shapeGamma = (1, 3, 1, 1)
#     shapeBeta = (1, 3, 1, 1)
    
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     gamma = np.random.uniform(0, 10, size=shapeGamma).astype(dtype)
#     beta = np.random.uniform(0, 10, size=shapeBeta).astype(dtype)
#     dout = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     dx = np.zeros(shapeX).astype(dtype)
#     dgamma = np.zeros(shapeGamma).astype(dtype)
#     dbeta = np.zeros(shapeBeta).astype(dtype)
#     dx_norm = np.zeros(shapeX).astype(dtype)
    
#     mean = np.mean(x, axis=(0, 2, 3)).reshape(1, 3, 1, 1)
#     var = np.var(x, axis=(0, 2, 3)).reshape(1, 3, 1, 1)

#     arr_x = tvm.nd.array(x)
#     arr_gamma = tvm.nd.array(gamma)
#     arr_beta = tvm.nd.array(beta)
#     arr_dout = tvm.nd.array(dout)
#     arr_dx = tvm.nd.array(dx)
#     arr_dgamma = tvm.nd.array(dgamma)
#     arr_dbeta = tvm.nd.array(dbeta)
#     arr_dx_norm = tvm.nd.array(dx_norm)
    
#     tvm_batch_norm_2d_backward = tvm_op.make_batchnorm2d_grad(shapeX, tgt, tgt_host, "batch_norm_2d_backward")
    
#     # input_mat, gamma, mean, var, dout, dx, dgamma, dbeta
#     tvm_batch_norm_2d_backward(arr_x, arr_gamma, arr_mean, arr_var, arr_dout, arr_dx, arr_dgamma, arr_dbeta)
    
    # np.testing.assert_allclose(batch_norm_2d_backward(dout, x, gamma, mean, var, 1e-5), arr_dx_norm.asnumpy(), rtol=1e-5)
    
    # dx, dgamma, dbeta = batch_norm_2d_backward(dout, cache)
    # np.testing.assert_allclose(dx, arr_dx.asnumpy(), rtol=1e-5)
    # np.testing.assert_allclose(dgamma, arr_dgamma.asnumpy(), rtol=1e-5)
    # np.testing.assert_allclose(dbeta, arr_dbeta.asnumpy(), rtol=1e-5)

# def flatten(x):
#     N = x.shape[0]
#     return x.reshape(N, -1)

# def test_flatten():
#     shapeX = (2, 3, 4, 4)
#     shapeY = (2, 3*4*4)
     
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     y = np.zeros(shapeY).astype(dtype)
    
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)

#     tvm_flatten = tvm_op.make_flatten(shapeX, tgt, tgt_host, "flatten")
#     tvm_flatten(arr_x, arr_y)
    
#     np.testing.assert_allclose(flatten(x), arr_y.asnumpy(), rtol=1e-5)

# def flatten_backward(dout, x):
#     return dout.reshape(x.shape)

# def test_flatten_backward():
#     shapeX = (2, 3, 4, 4)
#     shapeY = (2, 3*4*4)
     
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     dout = np.random.uniform(0, 10, size=shapeY).astype(dtype)
#     dx = np.zeros(shapeX).astype(dtype)
    
#     arr_x = tvm.nd.array(x)
#     arr_dout = tvm.nd.array(dout)
#     arr_dx = tvm.nd.array(dx)

#     tvm_flatten_backward = tvm_op.make_flatten_grad(shapeX, tgt, tgt_host, "flatten_backward")
#     tvm_flatten_backward(arr_dout, arr_x, arr_dx)
    
#     np.testing.assert_allclose(flatten_backward(dout, x), arr_dx.asnumpy(), rtol=1e-5)

# def np_conv2d_naive(x, w, padding=0, stride=1):
#     out = None
    
#     assert padding == 0, "padding is not supported yet"
#     assert stride == 1, "stride can only be 1 for now"
    
#     N, C, H, W = x.shape
#     M, _, R, S = w.shape
#     HO = int(1 + (H + 2*padding - R) / stride)
#     WO = int(1 + (W + 2*padding - S) / stride)
    
#     out = np.zeros((N, M, HO, WO))

#     for n in range(N):       
#         for m in range(M):   
#             for i in range(HO):
#                 for j in range(WO):
#                     for r in range(R): 
#                         for s in range(S):
#                             for c in range(C): 
#                                 out[n,m,i,j] += x[n, c, stride*i+r, stride*j+s] * w[m, c, r, s]
    
#     return out


# # im2col and np_conv2d are helper functions
# def im2col(X, filter_H, filter_W, padding, stride):
#     N, C, H, W = X.shape
#     # make all the inputs integers
#     N, C, H, W, filter_H, filter_W, padding, stride = \
#         int(N), int(C), int(H), int(W), int(filter_H), int(filter_W), int(padding), int(stride)
#     assert (H + 2 * padding - filter_H) % stride == 0
#     assert (W + 2 * padding - filter_W) % stride == 0
#     out_H = (H + 2 * padding - filter_H) // stride + 1
#     out_W = (W + 2 * padding - filter_W) // stride + 1

#     y_row_size = C * filter_H * filter_W
#     y_col_size = out_H * out_W
#     y_shape = (N, y_row_size, y_col_size)
#     Y = np.empty(y_shape, dtype = X.dtype)

#     for batch_index in range(N):
#         for col_index in range(y_col_size):
#             out_y = int(col_index / out_W)
#             out_x = int(col_index % out_W)
#             in_y = int(out_y * stride - padding)
#             in_x = int(out_x * stride - padding)
#             row_idx = 0
#             for c in range(0, C):
#                 for y in range(in_y, in_y + filter_H):
#                     for x in range(in_x, in_x + filter_W):
#                         if (x < 0 or x >= W or y < 0 or y >= H):
#                             Y[batch_index, row_idx, col_index] = 0
#                         else:
#                             Y[batch_index, row_idx, col_index] = X[batch_index, c, y, x]
#                             row_idx += 1
#     return Y

# def np_conv2d(X, Filter, padding=0, stride=1):
#     """Implement a conv2d as a matrix multiply after im2col."""
#     filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
#     N, C, H, W = X.shape
#     assert (H + 2 * padding - filter_H) % stride == 0
#     assert (W + 2 * padding - filter_W) % stride == 0
#     out_H = (H + 2 * padding - filter_H) // stride + 1
#     out_W = (W + 2 * padding - filter_W) // stride + 1

#     im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
#     filter_matrix = Filter.reshape(filter_outChannel, -1)
#     return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)

# def test_conv2d():
#     shapeX = (30, 30, 48, 48)
#     shapeF = (10, 30, 8, 8)
#     shapeY = (shapeX[0], shapeF[0], shapeX[2] - shapeF[2] + 1, shapeX[3] - shapeF[3] + 1)
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     f = np.random.uniform(0, 10, size=shapeF).astype(dtype)
#     y = np.zeros(shapeY).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_f = tvm.nd.array(f)
#     arr_y = tvm.nd.array(y)

#     # start = time()
#     conv2d = tvm_op.make_conv2d(shapeX, shapeF, tgt, tgt_host, "conv2d")
#     # conv2d(arr_x, arr_f, arr_y)
#     # tvm_taken = time() - start
#     # y = arr_y.asnumpy()
    
#     # start = time()
#     # y_np_naive = np_conv2d_naive(x, f)
#     # np_naive_taken = time() - start
#     # np.testing.assert_allclose(y_np_naive, y, rtol=1e-5)
    
#     # start = time()
#     # y_np_im2col = np_conv2d(x, f)
#     # np_im2col_taken = time() - start
#     # np.testing.assert_allclose(y_np_im2col, y, rtol=1e-5)
    
#     with open("exp/tvm_result.txt", "w") as f:
#         # f.write("Numpy Naive Taken time" + str(round(np_naive_taken, 2)) + "\n")
#         # f.write("Numpy im2col Taken time" + str(round(np_im2col_taken, 2)) + "\n")
#         # f.write("TVM Taken time: " + str(round(tvm_taken, 2)) + "\n")
#         dev = tvm.cpu(0)
#         f.write("TVM Evaluator time: " + str(round(conv2d.time_evaluator(conv2d.entry_name, dev, number=100)(arr_x, arr_f, arr_y).mean, 2)) + "\n")

# def test_conv2d_gpu():
#     dev = tvm.cuda(0)
#     shapeX = (30, 30, 100, 100)
#     shapeF = (10, 30, 8, 8)
#     shapeY = (shapeX[0], shapeF[0], shapeX[2] - shapeF[2] + 1, shapeX[3] - shapeF[3] + 1)
    
#     # X: [M, C, H, W] -> [H, W, C, M]
#     # F: [N, C, R, S] -> [R, S, C, N]
#     shapeX = shapeX[2], shapeX[3], shapeX[1], shapeX[0]
#     shapeF = shapeF[2], shapeF[3], shapeF[1], shapeF[0]
#     shapeY = shapeY[2], shapeY[3], shapeY[1], shapeY[0]
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     f = np.random.uniform(0, 10, size=shapeF).astype(dtype)
#     y = np.zeros(shapeY).astype(dtype)
#     arr_x = tvm.nd.array(x, dev)
#     arr_f = tvm.nd.array(f, dev)
#     arr_y = tvm.nd.array(y, dev)

#     start = time()
#     conv2d = tvm_op.make_conv2d_gpu(shapeX, shapeF, tgt, tgt_host, "conv2d")
#     conv2d(arr_x, arr_f, arr_y)
#     tvm_taken = time() - start
#     y = arr_y.asnumpy()
    
#     x = np.transpose(x, (3, 2, 0, 1))
#     f = np.transpose(f, (3, 2, 0, 1))
#     y = np.transpose(y, (3, 2, 0, 1))
    
#     start = time()
#     # y_np_im2col = np_conv2d(x, f)
#     np_im2col_taken = time() - start
    
#     # np.testing.assert_allclose(y_np_im2col, y, rtol=1e-5)
    
#     with open("tvm_result.txt", "w") as f:
#         # f.write("Numpy Naive Taken time" + str(round(np_im2col_taken, 2)) + "\n")
#         f.write("TVM GPU Taken time: " + str(round(tvm_taken, 2)) + "\n")
#         f.write("TVM GPU Evaluator time: " + str(round(conv2d.time_evaluator(conv2d.entry_name, dev, number=1)(arr_x, arr_f, arr_y).mean, 6)) + "\n")

# def np_conv2d_grad(dout, cache):
#     """
#     A naive implementation of the backward pass for a convolutional layer.

#     Inputs:
#     - dout: Upstream derivatives.
#     - cache: A tuple of (x, w, conv_param)

#     Returns a tuple of:
#     - dx: Gradient with respect to x
#     - dw: Gradient with respect to w
#     """
    
#     dx, dw = None, None
    
#     x, w, conv_param = cache
#     pad = conv_param['pad'] # always 0
#     assert pad == 0, "Current implementation only supports pad = 0"
#     stride = conv_param['stride'] # always 1
#     assert stride == 1, "Current implementation only supports stride = 1"
    
#     dx = np.zeros_like(x)
#     dw = np.zeros_like(w)
    
#     N, C, H, W = x.shape
#     M, _, R, S = w.shape
#     _, _, HO, WO = dout.shape
    
#     for n in range(N):      
#         for m in range(M):  
#             for i in range(HO):
#                 for j in range(WO):
#                     for r in range(R):
#                         for s in range(S):
#                             for c in range(C): 
#                                 dw[m,c,r,s] += x[n,c,stride*i+r,stride*j+s] * dout[n,m,i,j]

#     for n in range(N):      
#         for m in range(M):  
#             for i in range(HO):
#                 for j in range(WO):
#                     for r in range(R):
#                         for s in range(S):
#                             for c in range(C): 
#                                 dx[n,c,stride*i+r,stride*j+s] += w[m,c,r,s] * dout[n,m,i,j]
    
#     return dx, dw

# def test_conv2d_gradient_x():
#     # Define the shapes of the input tensor, filter, and output tensor
#     shapeX = (30, 3, 14, 14)
#     shapeF = (10, 3, 5, 5)
#     shapeY = (30, 10, 10, 10)
    
#     # Create a random input gradient tensor and initialize input tensor, filter, and output gradient tensor with random values
#     grad_y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     f = np.random.uniform(0, 10, size=shapeF).astype(dtype)
#     grad_x = np.zeros(shapeX).astype(dtype)
    
#     # Create TVM NDArray objects from numpy arrays
#     arr_grad_y = tvm.nd.array(grad_y)
#     arr_x = tvm.nd.array(x)
#     arr_f = tvm.nd.array(f)
#     arr_grad_x = tvm.nd.array(grad_x)
    
#     # Create a TVM operator
#     tvm_op_conv2d_grad_x = tvm_op.make_conv2d_grad_x(shapeX, shapeF, tgt, tgt_host, "conv2d_grad_x")
#     tvm_op_conv2d_grad_x(arr_grad_y, arr_x, arr_f, arr_grad_x)
#     grad_x = arr_grad_x.asnumpy()
    
#     # get gold result
#     gold_conv2d_grad_x = np_conv2d_grad(grad_y, (x, f, {'pad': 0, 'stride': 1}))[0]
    
#     # Compare the TVM result with the expected result using numpy's allclose function
#     np.testing.assert_allclose(gold_conv2d_grad_x, grad_x, rtol=1e-5)

# def test_conv2d_gradient_f():
#     # Define the shapes of the input tensor, filter, and output tensor
#     shapeX = (30, 3, 14, 14)
#     shapeF = (10, 3, 5, 5)
#     shapeY = (30, 10, 10, 10)
    
#     # Create a random input gradient tensor and initialize input tensor, filter, and output gradient tensor with random values
#     grad_y = np.random.uniform(0, 10, size=shapeY).astype(dtype)
#     x = np.random.uniform(0, 10, size=shapeX).astype(dtype)
#     f = np.random.uniform(0, 10, size=shapeF).astype(dtype)
#     grad_f = np.zeros(shapeF).astype(dtype)
    
#     # Create TVM NDArray objects from numpy arrays
#     arr_grad_y = tvm.nd.array(grad_y)
#     arr_x = tvm.nd.array(x)
#     arr_f = tvm.nd.array(f)
#     arr_grad_f = tvm.nd.array(grad_f)
    
#     # Create a TVM operator
#     tvm_op_conv2d_grad_f = tvm_op.make_conv2d_grad_f(shapeX, shapeF, tgt, tgt_host, "conv2d_grad_f")
#     tvm_op_conv2d_grad_f(arr_grad_y, arr_x, arr_f, arr_grad_f)
#     grad_f = arr_grad_f.asnumpy()
    
#     # get gold result
#     gold_conv2d_grad_f = np_conv2d_grad(grad_y, (x, f, {'pad': 0, 'stride': 1}))[1]
    
#     # Compare the TVM result with the expected result using numpy's allclose function
#     np.testing.assert_allclose(gold_conv2d_grad_f, grad_f, rtol=1e-5)

# def test_reduce_sum_axis_zero():
#     shape = (500, 200, 100)
#     to_shape = (200, 100)
#     x = np.random.uniform(-5, 5, shape).astype(dtype)
#     y = np.zeros(to_shape).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)

#     reduce_sum_axis_zero = tvm_op.make_reduce_sum_axis_zero(shape, tgt, tgt_host, "reduce_sum_axis_zero")
#     reduce_sum_axis_zero(arr_x, arr_y)
    
#     y = arr_y.asnumpy()
#     np.testing.assert_allclose(np.sum(x, axis=0), y, rtol=1e-5)

# def test_broadcast_to():
#     shape = (200, 300)
#     to_shape = (130, 200, 300)
#     x = np.random.uniform(-1, 1, shape).astype(dtype)
#     y = np.zeros(to_shape).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)
#     broadcast_to = tvm_op.make_broadcast_to(shape, to_shape, tgt, tgt_host, "broadcast_to")
#     broadcast_to(arr_x, arr_y)
#     y = arr_y.asnumpy()
#     np.testing.assert_allclose(np.broadcast_to(x, to_shape), y)

# def test_softmax():
#     shape = (400, 1000)
#     x = np.random.uniform(-5, 5, shape).astype(dtype)
#     y = np.zeros(shape).astype(dtype)
#     arr_x = tvm.nd.array(x)
#     arr_y = tvm.nd.array(y)
#     matrix_softmax = tvm_op.make_matrix_softmax(shape, tgt, tgt_host, "matrix_softmax")
#     matrix_softmax(arr_x, arr_y)
#     y = arr_y.asnumpy()
#     np.testing.assert_allclose(autodiff.softmax_func(x), y, rtol=1e-5)

def test_softmax_cross_entropy():
    shape = (400, 1000)
    y = np.random.uniform(-5, 5, shape).astype(dtype)
    y_ = np.random.uniform(-5, 5, shape).astype(dtype)
    out = np.zeros((1,)).astype(dtype)
    arr_y = tvm.nd.array(y)
    arr_y_ = tvm.nd.array(y_)
    arr_out = tvm.nd.array(out)
    matrix_softmax_cross_entropy = tvm_op.make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, "softmax_cross_entropy")
    matrix_softmax_cross_entropy(arr_y, arr_y_, arr_out)
    out = arr_out.asnumpy()
    
    # numpy calculation
    # cross_entropy = np.mean(
    #     -np.sum(y_ * np.log(autodiff.softmax_func(y)), axis=1), keepdims=True)
    
    # torch calculation
    import torch
    criterion = torch.nn.CrossEntropyLoss()
    cross_entropy = criterion(torch.from_numpy(y), torch.from_numpy(y_))
    np.testing.assert_allclose(cross_entropy, out, rtol=1e-5)
    