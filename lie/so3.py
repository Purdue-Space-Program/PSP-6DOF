import jax.numpy as jnp

def hat(w: jnp.ndarray) -> jnp.ndarray: # R3 -> so(3)
  return jnp.array([
    [0, -w[2], w[1]],
    [w[2], 0, -w[0]],
    [-w[1], w[0], 0]
  ])


def vee(W: jnp.ndarray) -> jnp.ndarray: # so(3) -> R3
  return jnp.array([W[2, 1], W[0, 2], W[1, 0]])

def exp(w: jnp.ndarray) -> jnp.ndarray:
  angle = jnp.linalg.norm(w)
  W = hat(w)
  safe_angle = jnp.where(angle < 1e-7, 1.0, angle)
  coeff1 = jnp.where(angle < 1e-7, 1.0 - angle**2/6.0, jnp.sin(safe_angle)/safe_angle)
  coeff2 = jnp.where(angle < 1e-7, 0.5 - angle**2/24.0, (1 - jnp.cos(safe_angle))/safe_angle**2)
  return jnp.eye(3) + coeff1*W + coeff2*(W @ W)

def log(R: jnp.ndarray) -> jnp.ndarray:
  angle = jnp.arccos(jnp.clip((jnp.trace(R) - 1) / 2, -1.0, 1.0))
  safe_sin = jnp.where(jnp.abs(angle) < 1e-7, 1.0, jnp.sin(angle))
  coeff = jnp.where(jnp.abs(angle) < 1e-7, 0.5 + angle**2/12.0, angle/(2*safe_sin))
  return vee(coeff * (R - R.T))

def adjoint(R: jnp.ndarray) -> jnp.ndarray: # adjoint representation
  return R