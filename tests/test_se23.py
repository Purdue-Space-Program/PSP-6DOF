import pytest
import jax.numpy as jnp
from lie.se23 import hat, vee, exp, log, adjoint, Qr, Ql, Jl_inv, Jr_inv

def test_hat_vee_roundtrip():
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3])
  assert jnp.allclose(vee(hat(xi)), xi, atol=1e-6)

def test_exp_is_valid_group_element():
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3])
  X = exp(xi)
  R = X[:3, :3]
  # rotation block should be valid SO(3)
  assert jnp.allclose(R.T @ R, jnp.eye(3), atol=1e-6)
  assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)
  # bottom rows should be correct
  assert jnp.allclose(X[3, :3], jnp.zeros(3), atol=1e-6)
  assert jnp.allclose(X[4, :3], jnp.zeros(3), atol=1e-6)
  assert jnp.allclose(X[3, 3], 1.0, atol=1e-6)
  assert jnp.allclose(X[4, 4], 1.0, atol=1e-6)

def test_exp_log_roundtrip():
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3])
  assert jnp.allclose(log(exp(xi)), xi, atol=1e-6)

def test_exp_log_roundtrip_small_angle():
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1e-9, 1e-9, 1e-9])
  assert jnp.allclose(log(exp(xi)), xi, atol=1e-6)

def test_exp_zero():
  xi = jnp.zeros(9)
  assert jnp.allclose(exp(xi), jnp.eye(5), atol=1e-6)

def test_log_identity():
  assert jnp.allclose(log(jnp.eye(5)), jnp.zeros(9), atol=1e-6)

def test_exp_no_nan_small_angle():
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1e-9, 1e-9, 1e-9])
  assert not jnp.any(jnp.isnan(exp(xi)))

def test_log_no_nan_small_angle():
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1e-9, 1e-9, 1e-9])
  X = exp(xi)
  assert not jnp.any(jnp.isnan(log(X)))

def test_Qr_zero():
  w = jnp.zeros(3)
  assert jnp.allclose(Qr(w), 0.5*jnp.eye(3), atol=1e-6)

def test_Qr_no_nan_small_angle():
  w = jnp.array([1e-9, 1e-9, 1e-9])
  assert not jnp.any(jnp.isnan(Qr(w)))

def test_Ql_no_nan_small_angle():
  w = jnp.array([1e-9, 1e-9, 1e-9])
  assert not jnp.any(jnp.isnan(Ql(w)))

def test_Jl_inv_inverse():
  from lie.se23 import Jl_inv
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3])
  # Jr_inv(-xi) = Jl_inv(xi), so Jl_inv @ Jr_inv(-(-xi)) should give I
  JlinvT = Jl_inv(xi)
  assert not jnp.any(jnp.isnan(JlinvT))
  assert JlinvT.shape == (9, 9)

def test_Jr_inv_is_Jl_inv_negated():
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3])
  assert jnp.allclose(Jr_inv(xi), Jl_inv(-xi), atol=1e-6)

def test_adjoint_shape():
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3])
  X = exp(xi)
  Ad = adjoint(X)
  assert Ad.shape == (9, 9)

def test_adjoint_no_nan():
  xi = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3])
  X = exp(xi)
  assert not jnp.any(jnp.isnan(adjoint(X)))