"""A library to take autodiff and execute a computation graph """
from __future__ import absolute_import

import numpy as np
import tvm
from . import tvm_op

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object call method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant.
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            # Mul by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = mul_byconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name


def Variable(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node


class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        output_val: output value of the node, modified in-place.
        compiled_func: compiled function that can be called on function inputs
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given output gradient, compute partial gradient to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """Given shapes of input nodes, compute shape of output node.

        Implementation note:
        It's simpler to treat shape of constants as (1,), so that constants can
        be stored as a numpy array too and you would need fewer special case
        handling.

        Parameters
        ----------
        node: node whose shape is being inferred.
        input_vals: shapes of input nodes.

        Returns
        -------
        A tuple representing the shape of output node.
        """
        raise NotImplementedError

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        """Compile the tvm function to native code for given input shapes.

        Parameters
        ----------
        node: node where the compute is done.
        input_shapes: shapes of input nodes.
        tgt: target device where computation is done, e.g. "llvm", "cuda", "arm" 
        tgt_host: target host where driver code is generated, e.g. "llvm"
               
        Returns
        -------
        A python function that you can directly call on op inputs and output.
        """
        raise NotImplementedError


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 2
        assert input_vals[0].shape == input_vals[1].shape
        compiled_func(input_vals[0], input_vals[1], output_val)  

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        return broadcast_rule(input_shapes[0], input_shapes[1])

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_elemwise_add(
            input_shapes[0], tgt, tgt_host, "elem_add")


class AddByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        compiled_func(input_vals[0], output_val)  

    def gradient(self, node, output_grad):
        return [output_grad]

    def infer_shape(self, node, input_shapes):
        shape_const = (1, )
        return broadcast_rule(shape_const, input_shapes[0])

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_elemwise_add_by_const(
            input_shapes[0], node.const_attr, tgt, tgt_host, "elem_add_const")

class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 2
        assert input_vals[0].shape == input_vals[1].shape
        compiled_func(input_vals[0], input_vals[1], output_val)  

    def gradient(self, node, output_grad):
        return [node.inputs[1] * output_grad, node.inputs[0] * output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        return broadcast_rule(input_shapes[0], input_shapes[1])

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_elemwise_mul(
            input_shapes[0], tgt, tgt_host, "elem_mul")

class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        compiled_func(input_vals[0], output_val)  

    def gradient(self, node, output_grad):
        return [node.const_attr * output_grad]

    def infer_shape(self, node, input_shapes):
        shape_const = (1, )
        return broadcast_rule(shape_const, input_shapes[0])

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_elemwise_mul_by_const(
            input_shapes[0], node.const_attr, tgt, tgt_host, "elem_mul_const")

class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A^T dY
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B^T, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B^T)^T=B dY^T, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        return [lhs_grad, rhs_grad]

    def infer_shape(self, node, input_shapes):
        # input_shapes is 2-dimensional
        assert(len(input_shapes[0]) == 2) and (len(input_shapes[1]) == 2)
        if (node.matmul_attr_trans_A == False) and (node.matmul_attr_trans_B == False):
            left = input_shapes[0]
            right = input_shapes[1]
        elif (node.matmul_attr_trans_A == False) and (node.matmul_attr_trans_B == True):
            left = input_shapes[0]
            right = (input_shapes[1][1], input_shapes[1][0])
        elif (node.matmul_attr_trans_A == True) and (node.matmul_attr_trans_B == False):
            left = (input_shapes[0][1], input_shapes[0][0])
            right = input_shapes[1]
        else:
            left = (input_shapes[0][1], input_shapes[0][0])
            right = (input_shapes[1][1], input_shapes[1][0])
        assert left[1] == right[0], "shape mismatch: %s, %s" % (str(left), str(right))
        return (left[0], right[1])
        
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_matrix_mul(
            input_shapes[0], node.matmul_attr_trans_A, input_shapes[1], node.matmul_attr_trans_B, tgt, tgt_host, "mat_mul"
        )

class Conv2dOp(Op):
    """We only consider the case where stride = 1 and padding = 0 for now."""
    def __call__(self, node_X, node_F):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, node_F]
        new_node.name = "Conv2d(%s,%s)" % (node_X.name, node_F.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        X, F = node.inputs
        X_grad = conv2d_gradient_x_op(X, F, output_grad)
        F_grad = conv2d_gradient_f_op(X, F, output_grad)
        return (X_grad, F_grad)

    def infer_shape(self, node, input_shapes):
        X_shape, F_shape = input_shapes
        N, C, H, W = X_shape
        M, C, R, S = F_shape
        out_shape = (N, M, H - R + 1, W - S + 1)
        return out_shape

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_conv2d(
            input_shapes[0], input_shapes[1], tgt, tgt_host, "conv2d"
        )
        
class Conv2dGradientXOp(Op):
    def __call__(self, node_X, node_F, output_grad):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, node_F, output_grad]
        new_node.name = "Conv2dGradientX(%s,%s)" % (node_X.name, node_F.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        # compiled_func(input_vals[0], input_vals[1], input_vals[2], output_val)
        x = input_vals[0].asnumpy()
        w = input_vals[1].asnumpy()
        dout = input_vals[2].asnumpy()
        pad = 0
        stride = 1
        output_val = np_conv2d_grad_x(dout, x, w, pad, stride)
        output_val = tvm.nd.array(output_val)
        
    def gradient(self, node, output_grad):
        return NotImplementedError
    
    def infer_shape(self, node, input_shapes):
        return input_shapes[0]
        
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_conv2d_grad_x(
            input_shapes[0], input_shapes[1], tgt, tgt_host, "conv2d_grad_x"
        )
        

class Conv2dGradientFOp(Op):
    def __call__(self, node_X, node_F, output_grad):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, node_F, output_grad]
        new_node.name = "Conv2dGradientF(%s,%s)" % (node_X.name, node_F.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], input_vals[1], input_vals[2], output_val)
        
    def gradient(self, node, output_grad):
        return NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[1]
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_conv2d_grad_f(
            input_shapes[0], input_shapes[1], tgt, tgt_host, "conv2d_grad_f"
        )

class Maxpool2dOp(Op):
    def __call__(self, node_X, pool_size, stride):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X]
        new_node.pool_size = pool_size
        new_node.stride = stride
        new_node.name = "Maxpool2d(%s)" % (node_X.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], output_val)
        
    def gradient(self, node, output_grad):
        X = node.inputs[0]
        X_grad = maxpool2d_gradient_op(X, output_grad, node.pool_size, node.stride)
        return [X_grad]
    
    def infer_shape(self, node, input_shapes):
        N, C, H, W = input_shapes[0]
        pool_size = node.pool_size
        stride = node.stride
        out_shape = (N, C, (H - pool_size) // stride + 1, (W - pool_size) // stride + 1)
        return out_shape
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_maxpool2d(
            input_shapes[0], node.pool_size, node.stride, tgt, tgt_host, "maxpool2d"
        )
    
class Maxpool2dGradientOp(Op):
    def __call__(self, node_X, output_grad, pool_size, stride):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, output_grad]
        new_node.pool_size = pool_size
        new_node.stride = stride
        new_node.name = "Maxpool2dGradient(%s)" % (node_X.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        # compiled_func(input_vals[0], input_vals[1], output_val)
        
        # convert values to numpy
        input_vals = [val.asnumpy() for val in input_vals]
        
        output_val = np_maxpool2d_backward(input_vals[1], input_vals[0], node.pool_size, node.stride)
        
        # convert numpy output_val to tvm NDArray
        output_val = tvm.nd.array(output_val)
        
    def gradient(self, node, output_grad):
        return NotImplementedError
    
    def infer_shape(self, node, input_shapes):
        return input_shapes[0]
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        # return tvm_op.make_maxpool2d_grad(
        #     input_shapes[0], node.pool_size, node.stride, tgt, tgt_host, "maxpool2d_grad"
        # )
        pass
        
class BatchNorm2dOp(Op):
    def __call__(self, node_X, gamma, beta):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, gamma, beta]
        new_node.name = "BatchNorm2d(%s)" % (node_X.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], input_vals[1], input_vals[2], output_val)
         
    def gradient(self, node, output_grad):
        X = node.inputs[0]
        gamma = node.inputs[1]
        beta = node.inputs[2]
        X_grad = batchnorm2d_gradient_x_op(X, gamma, output_grad)
        gamma_grad = batchnorm2d_gradient_gamma_op(X, gamma, output_grad)
        beta_grad = batchnorm2d_gradient_beta_op(X, gamma, output_grad)
        return [X_grad, gamma_grad, beta_grad]
        
    def infer_shape(self, node, input_shapes):
        return input_shapes[0]
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_batchnorm2d(
            input_shapes[0], tgt, tgt_host, "batchnorm2d"
        )
        
class BatchNorm2dGradientXOp(Op):
    def __call__(self, node_X, gamma, output_grad):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, gamma, output_grad]
        new_node.name = "BatchNorm2dGradientX(%s)" % (node_X.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        # compiled_func(input_vals[0], input_vals[1], input_vals[2], input_vals[3], input_vals[4], output_val)
        
        input_vals = [val.asnumpy() for val in input_vals]
        
        X = input_vals[0]
        gamma = input_vals[1]
        output_grad = input_vals[2]
        X_grad, _, _ = np_batch_norm_2d_backward(output_grad, X, gamma)
        
        output_val = tvm.nd.array(X_grad)
        
    def gradient(self, node, output_grad):
        return NotImplementedError
    
    def infer_shape(self, node, input_shapes):
        return input_shapes[0]
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        pass
        # return tvm_op.make_batchnorm2d_grad_x(
        #     input_shapes[0], tgt, tgt_host, "batchnorm2d_grad_x"
        # )
        
class BatchNorm2dGradientGammaOp(Op):
    def __call__(self, node_X, gamma, output_grad):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, gamma, output_grad]
        new_node.name = "BatchNorm2dGradientGamma(%s)" % (node_X.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        # compiled_func(input_vals[0], input_vals[1], input_vals[2], input_vals[3], input_vals[4], output_val)
        
        input_vals = [val.asnumpy() for val in input_vals]
        
        X = input_vals[0]
        gamma = input_vals[1]
        output_grad = input_vals[2]
        _, gamma_grad, _ = np_batch_norm_2d_backward(output_grad, X, gamma)
        
        output_val = tvm.nd.array(gamma_grad)
        
    def gradient(self, node, output_grad):
        return NotImplementedError
    
    def infer_shape(self, node, input_shapes):
        return (1, input_shapes[0][1], 1, 1)
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        pass
        # return tvm_op.make_batchnorm2d_grad_gamma(
        #     input_shapes[0], tgt, tgt_host, "batchnorm2d_grad_gamma"
        # )
        
class BatchNorm2dGradientBetaOp(Op):
    def __call__(self, node_X, gamma, output_grad):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, gamma, output_grad]
        new_node.name = "BatchNorm2dGradientBeta(%s)" % (node_X.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        # compiled_func(input_vals[0], input_vals[1], input_vals[2], input_vals[3], input_vals[4], output_val)
        
        input_vals = [val.asnumpy() for val in input_vals]
        
        X = input_vals[0]
        gamma = input_vals[1]
        output_grad = input_vals[2]
        _, _, beta_grad = np_batch_norm_2d_backward(output_grad, X, gamma)
        
        output_val = tvm.nd.array(beta_grad)
        
    def gradient(self, node, output_grad):
        return NotImplementedError
    
    def infer_shape(self, node, input_shapes):
        return (1, input_shapes[0][1], 1, 1)
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        pass
        # return tvm_op.make_batchnorm2d_grad_beta(
        #     input_shapes[0], tgt, tgt_host, "batchnorm2d_grad_beta"
        # )
    
class FlattenOp(Op):
    def __call__(self, node_X, CHW):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X]
        new_node.CHW = CHW
        new_node.name = "Flatten(%s)" % (node_X.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], output_val)
        
    def gradient(self, node, output_grad):
        node_X = node.inputs[0]
        X_grad = flatten_gradient_op(node_X, node.CHW, output_grad)
        return [X_grad]
    
    def infer_shape(self, node, input_shapes):
        N, C, H, W = input_shapes[0]
        return (N, C*H*W)
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_flatten(
            input_shapes[0], tgt, tgt_host, "flatten"
        )

class FlattenGradientOp(Op):
    def __call__(self, node_X, CHW, node_output_grad):
        new_node = Op.__call__(self)
        new_node.inputs = [node_X, node_output_grad]
        new_node.name = "FlattenGradient(%s)" % (node_X.name)
        new_node.CHW = CHW
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], input_vals[1], output_val)
        
    def gradient(self, node, output_grad):
        return NotImplementedError
    
    def infer_shape(self, node, input_shapes):
        N = input_shapes[0][0]
        C, H, W = node.CHW
        return [N, C, H, W]
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        N = input_shapes[0][0]
        C, H, W = node.CHW
        shapeX = (N, C, H, W)
        return tvm_op.make_flatten_grad(
            shapeX, tgt, tgt_host, "flatten_grad"
        )
        
class Concat1dOp(Op):
    """the inputs are always of shape [1, L]"""
    def __call__(self, node_input, node_hidden, axis):
        new_node = Op.__call__(self)
        new_node.inputs = [node_input, node_hidden]
        new_node.axis = axis
        assert axis == 1, "Currently concat1dOp only supports axis=1"
        new_node.name = "Concat1d(%s, %s)" % (node_input.name, node_hidden.name)
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        # use numpy directly, it's not slower than TVM
        input_vals = [val.asnumpy() for val in input_vals]
        output_val = np.concatenate(input_vals, axis=node.axis)
        
    def gradient(self, node, output_grad):
        node_input = node.inputs[0]
        node_hidden = node.inputs[1]
        input_grad = concat1d_gradient_input_op(node_input, node_hidden, node.axis, output_grad)
        hidden_grad = concat1d_gradient_hidden_op(node_input, node_hidden, node.axis, output_grad)
        return [input_grad, hidden_grad]
    
    def infer_shape(self, node, input_shapes):
        return (1, sum([shape[1] for shape in input_shapes]))
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return None
    
class Concat1dGradientInputOp(Op):
    def __call__(self, node_input, node_hidden, axis, node_output_grad):
        new_node = Op.__call__(self)
        new_node.inputs = [node_input, node_hidden, node_output_grad]
        new_node.name = "Concat1dGradientInput(%s, %s)" % (node_input.name, node_hidden.name)
        new_node.axis = axis
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        # use numpy directly, it's not slower than TVM
        input_vals = [val.asnumpy() for val in input_vals]
        output_val = np.split(output_val, [input_vals[0].shape[1], output_val.shape[1]], axis=1)[0]

    def gradient(self, node, output_grad):
        return NotImplementedError
    
    def infer_shape(self, node, input_shapes):
        return input_shapes[0]
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return None
    
class Concat1dGradientHiddenOp(Op):
    def __call__(self, node_input, node_hidden, axis, node_output_grad):
        new_node = Op.__call__(self)
        new_node.inputs = [node_input, node_hidden, node_output_grad]
        new_node.name = "Concat1dGradientHidden(%s, %s)" % (node_input.name, node_hidden.name)
        new_node.axis = axis
        return new_node
    
    def compute(self, node, input_vals, output_val, compiled_func):
        # use numpy directly, it's not slower than TVM
        input_vals = [val.asnumpy() for val in input_vals]
        output_val = np.split(output_val, [input_vals[0].shape[1], output_val.shape[1]], axis=1)[1]

    def gradient(self, node, output_grad):
        return NotImplementedError
    
    def infer_shape(self, node, input_shapes):
        return input_shapes[1]
    
    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return None


class PlaceholderOp(Op):
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert False, "placeholder %s values provided by feed_dict" % node.name

    def gradient(self, node, output_grad):
        return None

    def infer_shape(self, node, input_shapes):
        assert False, "placeholder %s shape provided by feed_shape" % node.name

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return None

class ZerosLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.zeros(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        output_val.copyfrom(
            np.zeros(input_vals[0].shape, dtype = input_vals[0].dtype))

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        return input_shapes[0]

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return None


class OnesLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.ones(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        output_val.copyfrom(
            np.ones(input_vals[0].shape, dtype = input_vals[0].dtype))

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        # different
        return input_shapes[0]

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return None


class ReduceSumAxisZeroOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceSumAxisZero(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        compiled_func(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        assert(len(input_shapes)==1)
        if len(input_shapes[0]) == 1:
            return (1,)
        return input_shapes[0][1:]

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_reduce_sum_axis_zero(
            input_shapes[0], tgt, tgt_host, "reduce_sum_axis_zero")


class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents np.broadcast_to(node_A, node_B.shape).
        Only support axis=0. e.g. (3,4)->(2,3,4) to make gradient simple.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastTo(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert(len(input_vals)==2)
        compiled_func(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        grad_A = reducesumaxiszero_op(output_grad)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        return broadcast_rule(input_shapes[0], input_shapes[1])

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_broadcast_to(
            input_shapes[0], input_shapes[1], tgt, tgt_host, "broadcast_to"
        )

def softmax_func(y):
    """Numerically stable softmax."""
    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax


class SoftmaxCrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "SoftmaxXEntropy(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        compiled_func(y, y_, output_val)

    def gradient(self, node, output_grad):
        grad_A_temp = softmax_op(node.inputs[0]) + -1 * node.inputs[1]
        grad_A = grad_A_temp * broadcastto_op(output_grad, grad_A_temp)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        return (1, )

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_matrix_softmax_cross_entropy(
            input_shapes[0], tgt, tgt_host, "softmax_cross_entropy"
        )

class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Softmax(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 1
        compiled_func(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        # Do not directly use SoftmaxOp, use SoftmaxCrossEntropyOp instead.
        # Not allowing taking 2nd derivative of SoftmaxCrossEntropyOp.
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_matrix_softmax(
            input_shapes[0], tgt, tgt_host, "softmax"
        )


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Relu(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_relu(
            input_shapes[0], tgt, tgt_host, "relu"
        )


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        compiled_func(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_relu_gradient(
            input_shapes[0], tgt, tgt_host, "relu_grad"
        )

# Create global singletons of operators.

# basic
add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
# cnn
conv2d_op = Conv2dOp()
conv2d_gradient_x_op = Conv2dGradientXOp()
conv2d_gradient_f_op = Conv2dGradientFOp()
maxpool2d_op = Maxpool2dOp()
maxpool2d_gradient_op = Maxpool2dGradientOp()
batchnorm2d_op = BatchNorm2dOp()
batchnorm2d_gradient_x_op = BatchNorm2dGradientXOp()
batchnorm2d_gradient_gamma_op = BatchNorm2dGradientGammaOp()
batchnorm2d_gradient_beta_op = BatchNorm2dGradientBetaOp()
flatten_op = FlattenOp()
flatten_gradient_op = FlattenGradientOp()
# rnn
concat1d_op = Concat1dOp()
concat1d_gradient_input_op = Concat1dGradientInputOp()
concat1d_gradient_hidden_op = Concat1dGradientHiddenOp()
# common
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
reducesumaxiszero_op = ReduceSumAxisZeroOp()
broadcastto_op = BroadcastToOp()
softmaxcrossentropy_op = SoftmaxCrossEntropyOp()
softmax_op = SoftmaxOp()
relu_op = ReluOp()
relu_gradient_op = ReluGradientOp()


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""
    def __init__(self, eval_node_list, ctx=None):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        ctx: runtime DLContext, default is None which means np.ndarray on cpu
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to tvm.nd.array allocated for node
        node_to_compiled_func: dict from node to compiled func for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
        self.eval_node_list = eval_node_list
        self.ctx = ctx
        if self.ctx == tvm.cpu(0):
            self.tgt = "llvm"
            self.tgt_host="llvm"
        else:
            assert False, "non-CPU context not yet supported"
        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.node_to_compiled_func = None
        self.feed_shapes = None

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.op.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        self.node_to_shape_map = {}
        for node in self.topo_order:
            if node in feed_shapes:
                self.node_to_shape_map[node] = feed_shapes[node]
                continue
            input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
            self.node_to_shape_map[node] = node.op.infer_shape(node, input_shapes)

    def memory_plan(self, feed_shapes):
        """Allocates tvm.nd.array for every node except feed_dict nodes.

        Implementation note:
        Easy Option: Alloc a tvm.nd.array per node that persists across run()

        Use self.node_to_arr_map to store node->tvm.nd.array mapping
        to allow mapping to persist across multiple executor.run().

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        self.node_to_arr_map = {}
        for node in self.topo_order:
            # input nodes have no memory allocation need
            if node in feed_shapes:
                continue
            # we need to allocate memory for this node
            self.node_to_arr_map[node] = tvm.nd.empty(
                self.node_to_shape_map[node], dtype="float32"
            )
            

    def compile_funcs(self, feed_shapes):
        """Compile tvm ops to native code.

        Must be called after infer_shape(...) since op compilation requires
        knowledge of tensor shapes.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        self.node_to_compiled_func = {}
        for node in self.topo_order:
            if node in feed_shapes:
                continue
            input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
            self.node_to_compiled_func[node] = node.op.compiled_func(
                node, input_shapes, self.tgt, self.tgt_host
            )
        

    def run(self, feed_dict, convert_to_numpy_ret_vals=False, return_hidden=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array.
        return_hidden: whether to return hidden states of RNN.

        Returns
        -------
        A list of values for nodes in eval_node_list. tvm.nd.array or np.ndarray.
        """
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        node_to_val_map = {}
        for node, value in feed_dict.items():
            assert isinstance(value, tvm.nd.NDArray), "feed_dict value type not supported"    
            node_to_val_map[node] = value

        # collect shapes for all placeholders
        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        # infer shape if feed_shapes changed since last run
        # 1. first time call run() on training data
        # 2. call run() on test data after training
        if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            self.memory_plan(feed_shapes)
            self.compile_funcs(feed_shapes)

        ###########################################################
        ###                                                     ###
        ### This is the most important part of the run function ###
        ###                                                     ###
        ###########################################################
        
        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:
            if node in node_to_val_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                continue
            input_vals = [node_to_val_map[n] for n in node.inputs]
            node_val = self.node_to_arr_map[node]
            # node_val is modified in-place
            node.op.compute(
                node, input_vals, node_val, self.node_to_compiled_func[node])
            node_to_val_map[node] = node_val
        # Collect node values.
        if convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]
        
        # for node in self.eval_node_list:
        #     print(node)
        
        return [node_to_val_map[n] for n in self.eval_node_list]


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##################
# Helper Methods #
##################


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def broadcast_rule(shape_a, shape_b):
    """Return output shape of broadcast shape_a, shape_b.
    e.g. broadcast_rule((3,2), (4,3,2))
    returns output_shape = (4,3,2)

    Check out explanations and more examples at
    https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html
    http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    """
    assert(isinstance(shape_a, tuple))
    assert(isinstance(shape_b, tuple))
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in range(len_diff):
        # pad with leading 1s
        shorter_shape = (1,) + shorter_shape
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in range(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
            or (shorter_shape[i] == 1) \
            or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)

def np_batch_norm_2d_backward(dout, x, gamma, eps=1e-5):
    # Compute the batch size
    N, C, H, W = x.shape
    
    # compute the mean and var
    mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var = np.mean((x - mean)**2, axis=(0, 2, 3), keepdims=True)
    
    # Compute the standard deviation and inverse of the standard deviation
    std = np.sqrt(var + eps)
    istd = 1.0 / std
    
    # Compute the normalized input
    x_norm = (x - mean) / std
    
    # Compute the gradients with respect to gamma and beta
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    
    # Compute the gradient with respect to the input
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * (-0.5) * istd**3, axis=(0, 2, 3), keepdims=True)
    dmean = np.sum(dx_norm * (-istd), axis=(0, 2, 3), keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=(0, 2, 3), keepdims=True)
    dx = dx_norm * istd + dvar * 2.0 * (x - mean) / (N * H * W) + dmean / (N * H * W)
    
    return dx, dgamma, dbeta

def np_maxpool2d_backward(grad_output, x, pool_size=2, stride=2):
    N, C, H, W = x.shape
    _, _, HO, WO = grad_output.shape
    grad_input = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(HO):
                for j in range(WO):
                    start_i = i * stride
                    start_j = j * stride
                    patch = x[n, c, start_i:start_i+pool_size, start_j:start_j+pool_size]
                    max_index = np.unravel_index(np.argmax(patch), patch.shape)
                    grad_input[n, c, start_i+max_index[0], start_j+max_index[1]] += grad_output[n, c, i, j]
    return grad_input

def np_conv2d_grad_x(dout, x, w, pad, stride):
    dx, dw = None, None
    
    assert pad == 0, "Current implementation only supports pad = 0"
    assert stride == 1, "Current implementation only supports stride = 1"
    
    dx = np.zeros_like(x)
    
    N, C, H, W = x.shape
    M, _, R, S = w.shape
    _, _, HO, WO = dout.shape
                  
    # for n in range(N):      
    #     for m in range(M):  
    #         for i in range(HO):
    #             for j in range(WO):
    #                 for r in range(R):
    #                     for s in range(S):
    #                         for c in range(C): 
    #                             dx[n,c,stride*i+r,stride*j+s] += w[m,c,r,s] * dout[n,m,i,j]
                                
    for n in range(N):
        for m in range(M):
            for i in range(HO):
                for j in range(WO):
                    h1 = i * stride
                    h2 = i * stride + R
                    w1 = j * stride
                    w2 = j * stride + S
                    dx[n, :, h1:h2, w1:w2] += w[m,:,:,:] * dout[n,m,i,j]
    
    return dx

