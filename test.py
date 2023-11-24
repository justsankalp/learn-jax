import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
print(jax.default_backend())
#damn it that's enough for today :)