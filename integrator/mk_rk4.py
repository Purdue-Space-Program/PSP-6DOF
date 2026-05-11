import jax.numpy as jnp
import lie.se23 as se23
from typing import Callable

def step(f: Callable[[jnp.ndarray, float], jnp.ndarray], X: jnp.ndarray, t: float, dt: float) -> jnp.ndarray:
  # rk4 coeff
  # c1=0, c2=1/2, c3=1/2, c4=1
  # d1=0, d2=0, d3=1/4, d4=1/2
  # m1=2, m2=2, m3=-1

  I1 = k1 = f(X, t)

  u2 = dt/2 * k1
  u2_corr = u2
  k2 = f(se23.exp(u2_corr) @ X, t + dt/2)

  u3 = dt/2 * k2
  u3_corr = u3 + (dt/2)/6 * se23.lie_bracket(I1, u3)
  k3 = f(se23.exp(u3_corr) @ X, t + dt/2)

  u4 = dt * k3
  u4_corr = u4 + (dt)/6 * se23.lie_bracket(I1, u4)
  k4 = f(se23.exp(u4_corr) @ X, t + dt)

  I2 = (2*(k2 - I1) + 2*(k3 - I1) - (k4 - I1)) / dt

  v = dt/6 * (k1 + 2*k2 + 2*k3 + k4)
  v_corr = v + dt/4 * se23.lie_bracket(I1, v) + dt**2/24 * se23.lie_bracket(I2, v)

  return se23.exp(v_corr) @ X
