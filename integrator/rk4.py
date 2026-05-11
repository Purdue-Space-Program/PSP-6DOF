import jax.numpy as jnp
from typing import Callable

def step(f: Callable[[jnp.ndarray, float], jnp.ndarray], X: jnp.ndarray, t: float, dt: float) -> jnp.ndarray:
  k1 = f(X, t)
  k2 = f(X + dt/2 * k1, t + dt/2)
  k3 = f(X + dt/2 * k2, t + dt/2)
  k4 = f(X + dt * k3, t + dt)
  return X + dt/6 * (k1 + 2*k2 + 2*k3 + k4)