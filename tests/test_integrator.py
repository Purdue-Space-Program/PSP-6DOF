import pytest
import jax.numpy as jnp
from lie import se23
from integrator import mk_rk4, rk4

def constant_dynamics(X, t):
  """Constant twist dynamics — pure rotation around z axis."""
  return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

def test_mk_rk4_stays_on_manifold():
  """MK-RK4 should stay on SE2(3) manifold."""
  X = jnp.eye(5)
  for _ in range(100):
    X = mk_rk4.step(constant_dynamics, X, 0.0, 0.01)
  R = X[:3, :3]
  assert jnp.allclose(R.T @ R, jnp.eye(3), atol=1e-6)
  assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)

def test_rk4_drifts_off_manifold():
  """Standard RK4 should drift off SE2(3) manifold over time."""
  def flat_dynamics(X, t):
    xi = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return se23.hat(xi)  # return as matrix since rk4 adds directly
  X = jnp.eye(5)
  for _ in range(100):
    X = rk4.step(flat_dynamics, X, 0.0, 0.01)
  R = X[:3, :3]
  # orthogonality should be violated
  assert not jnp.allclose(R.T @ R, jnp.eye(3), atol=1e-6)

def test_mk_rk4_identity_dynamics():
  """Zero dynamics should return identity."""
  X = jnp.eye(5)
  X_next = mk_rk4.step(lambda X, t: jnp.zeros(9), X, 0.0, 0.01)
  assert jnp.allclose(X_next, jnp.eye(5), atol=1e-6)

def test_mk_rk4_single_step_no_nan():
  """Single step should not produce NaN."""
  X = jnp.eye(5)
  X_next = mk_rk4.step(constant_dynamics, X, 0.0, 0.01)
  assert not jnp.any(jnp.isnan(X_next))