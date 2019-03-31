"""Architec controls architecture of cell by computing gradients of alphas
"""
import copy
import tensorflow as tf
from utils import *


class Architect():
    """Compute gradients of alphas"""
    def __init__(self, model):
        """
        Args:
            model
        """
        self.model = model
        self.v_model = copy.deepcopy(model)

        self.var_alphas = [vars for vars in self.model.trainable_variables if 'arch_kernel' in vars.name]
        self.var_weights = [vars for vars in self.model.trainable_variables if 'arch_kernel' not in vars.name]
        self.var_alphas_v = [vars for vars in self.v_model.trainable_variables if 'arch_kernel' in vars.name]
        self.var_weights_v = [vars for vars in self.v_model.trainable_variables if 'arch_kernel' not in vars.name]

    def virtual_step(self, train_x, train_y, weights_optimizer, global_step):

        grads = grad(self.model, train_x, train_y, self.var_weights)
        weights_optimizer.apply_gradients(zip(grads, self.var_weights_v), global_step)

    def unrolled_backward(self, train_x, train_y, val_x, val_y, xi, weights_optimizer, global_step):

        self.virtual_step(train_x, train_y, weights_optimizer, global_step)
        grad_var_weights_v = grad(self.v_model, val_x, val_y, self.var_weights_v)
        grad_var_alphas_v = grad(self.v_model, val_x, val_y, self.var_alphas_v)

        hessian = self.compute_hessian(grad_var_weights_v, train_x, train_y)

        for grads, hess in zip(grad_var_alphas_v, hessian):
            grads -= xi * hess

        return grad_var_alphas_v

    def compute_hessian(self, grad_var_weights_v, train_x, train_y):

        norm = tf.norm(tf.concat([tf.reshape(grads, (1, -1)) for grads in grad_var_weights_v], axis=1))
        eps = 0.01 / norm
        # w+ = w + eps * dw'
        for weights, grads in zip(self.var_weights, grad_var_weights_v):
            tf.assign_add(weights, eps * grads)
        grads_alphas_pos = grad(self.model, train_x, train_y, self.var_alphas)

        # w- = w - eps * dw'
        for weights, grads in zip(self.var_weights, grad_var_weights_v):
            tf.assign_sub(weights, 2 * eps * grads)
        grads_alphas_neg = grad(self.model, train_x, train_y, self.var_alphas)

        # recover w
        for weights, grads in zip(self.var_weights, grad_var_weights_v):
            tf.assign_add(weights, eps * grads)
        hessian = [(p-n) / 2. * eps for p, n in zip(grads_alphas_pos, grads_alphas_neg)]

        return hessian


















