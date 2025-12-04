import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import numpy as np
from time import time
from functools import partial

# This only works on startup!
from jax import config
config.update("jax_enable_x64", True)

from mms import apply_mms,get_BC_function
from quadrature import QuadratureMethod

key = jax.random.PRNGKey(42)

############################ PROBLEM ####################################

gdim = 2
N = 25
D = 3
NMC = 1000

QM = QuadratureMethod(gdim, seed=42)

if gdim == 1:
    uex_str = 'sin(10*pi*x[0])*x[0]'
    uex_str = 'sin(2*pi*x[0])'
    xtest = np.linspace(0, 1, 10000).reshape((1,-1))
elif gdim == 2:
    xx = np.linspace(0, 1, 100)
    X,Y = np.meshgrid(xx,xx)
    xtest = np.vstack([X.flatten(), Y.flatten()])

    uex_str = 'sin(10*pi*x[0])*x[0]*x[1]*sin(4*pi*x[1]) + exp(-4*((x[0]-0.5)**2 + (x[1]-0.5)**2))*16*x[0]*(1-x[0])*x[1]*(1-x[1])'
else:
    raise NotImplementedError

problem_data = apply_mms(uex_str)
u_ex = eval(problem_data['u_ex'])
f    = eval(problem_data['f'])

bc_func = get_BC_function(gdim)

xtrain,weights = QM.MC(NMC)

############################ MODEL AND LOSS FUNCTION ####################################

class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, key, N, D):
        if D < 2: raise NotImplementedError
        nn_dimensions = [[gdim, N]] + [[N, N] for i in range(D-2)] + [[N,1]]
        n_layers = len(nn_dimensions)
        keys = jax.random.split(key, n_layers)
        self.layers = [eqx.nn.Linear(nn_dimensions[i][0], nn_dimensions[i][1], key=keys[i]) for i in range(n_layers-1)]
        self.layers.append(eqx.nn.Linear(nn_dimensions[-1][0], "scalar", use_bias=False, key=keys[-1]))
        self.init_linear_weight(jax.nn.initializers.glorot_uniform, key)

    def init_linear_weight(self, init_fn, key):
        get_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m) if isinstance(x, eqx.nn.Linear)]
        weights = get_weights(self)
        new_weights = [init_fn(subkey, weight.shape) for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
        self = eqx.tree_at(get_weights, self, new_weights)

    def __call__(self, x):
        phi = bc_func(x)
        for layer in self.layers[:-1]:
            x = jax.nn.sigmoid(layer(x))
        return (self.layers[-1](x)*phi).squeeze()

    def u(self, x):
        return jax.vmap(self, in_axes=1, out_axes=0)(x).squeeze()

    def du(self, x):
        return jax.vmap(jax.jacfwd(self), in_axes=1, out_axes=0)(x).T

    def hess(self, x):
        return jax.vmap(jax.jacfwd(jax.jacfwd(self)), in_axes=1, out_axes=0)(x).T

    def split_eval(self, *x):
        return self(jnp.array([*x]))

    def laplacian_scalar(self, x):
        return sum(jax.jacfwd(jax.jacfwd(self.split_eval,i),i)(*x) for i in range(gdim))

    def laplacian(self, x):
        return sum(jax.vmap(jax.jacfwd(jax.jacfwd(self.split_eval,i),i))(*x) for i in range(gdim))

    def residual(self, x):
        return -self.laplacian(x) - f(x)

    def residual_scalar(self, x):
        return -self.laplacian_scalar(x) - f(x)

    # computes the gradient of the residual wrt theta
    # NOTE: the output is a pytree!
    @eqx.filter_grad
    def residual_dtheta_scalar(self, x):
        return self.residual_scalar(x)

    # vectorised gradient (wrt theta) computation (OPTION 1)
    # NOTE: the output is a single pytree with 3D tensor nodes!
    def residual_dtheta_pytree(self, x):
        return jax.vmap(self.residual_dtheta_scalar, in_axes=1)(x)

    # vectorised gradient (wrt theta) computation (OPTION 2)
    # NOTE: the output is now a 2D array of shape (n_x_coordinates, len(theta))
    def residual_dtheta(self, x):
        flattened_gradient_dtheta = lambda x : jax.flatten_util.ravel_pytree(self.residual_dtheta_scalar(x))[0]
        return jax.vmap(flattened_gradient_dtheta, in_axes=1)(x)

model = NeuralNetwork(key, N, D)

# vals is a 1D jax.numpy.array containing all NN weights and biases
# Can reconstruct the NN from the vals by using unravel_fun
vals,unravel_fun = jax.flatten_util.ravel_pytree(model)
original_model = unravel_fun(vals)

dNN_dtheta_pytree = model.residual_dtheta_pytree(xtrain)
dNN_dtheta = model.residual_dtheta(xtrain)

# example of recovering pytree from d_NN/d_theta at x_0
i = 0 # first coordinate of xtrain
dNN_dtheta_xi = unravel_fun(dNN_dtheta[i]) # pytree with same structure as model

optim = optax.adam(1.0e-2)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

# must return the loss function value and its gradient wrt theta
def loss_value_and_grad(model):
    raise NotImplementedError

@eqx.filter_jit
def epoch_update(model, loss_value, loss_grad, opt_state):
    error = abs(model.u(xtest)-u_ex(xtest)).max() # exact error in inf norm
    # take optimiser step
    updates, opt_state = optim.update(loss_grad, opt_state, model)
    model = eqx.apply_updates(model, updates)

    # compute gradient norm
    flat_model_grad = jax.flatten_util.ravel_pytree(loss_grad)[0]
    grad_norm = jnp.linalg.norm(flat_model_grad)

    return loss_value, grad_norm, error, model, opt_state

def train_epoch(model, opt_state):
    loss_value, loss_grad = loss_value_and_grad(model) # loss_grad must be a pytree
    return epoch_update(model, loss_value, loss_grad, opt_state)

niter = 10**4
losses = []
grad_norms = []
errors = []
tic = time()
for epoch in range(niter+1):
    xtrain, weights = QM.MC(NMC)

    loss,grad_norm,error,model,opt_state = train_epoch(model, opt_state)

    losses.append(loss)
    errors.append(error)
    grad_norms.append(grad_norm)

    if epoch % 100 == 0:
        print('Epoch %4d, error: %.6e, gradient norm: %.6e, loss: %.6e' % (epoch, error, grad_norm, loss), flush=True)

print("Training time: ", time()-tic)
