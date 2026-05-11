import pytest
import jax.numpy as jnp
from lie.so3 import hat, vee, exp, log, adjoint, Jl, Jl_inv, Jr, Jr_inv

def test_hat_vee_roundtrip():
  w = jnp.array([1.0, 2.0, 3.0])
  assert jnp.allclose(vee(hat(w)), w)

def test_exp_log_roundtrip():
  w = jnp.array([0.1, 0.2, 0.3])
  assert jnp.allclose(log(exp(w)), w)

def test_exp_is_rotation_matrix():
  w = jnp.array([0.1, 0.2, 0.3])
  R = exp(w)
  assert jnp.allclose(R.T @ R, jnp.eye(3), atol=1e-6)
  assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)

def test_hat_is_skew_symmetric():
  w = jnp.array([1.0, 2.0, 3.0])
  W = hat(w)
  assert jnp.allclose(W, -W.T)

def test_exp_small_angle():
  w = jnp.array([1e-9, 1e-9, 1e-9])
  R = exp(w)
  assert not jnp.any(jnp.isnan(R))
  assert jnp.allclose(R.T @ R, jnp.eye(3), atol=1e-6)

def test_exp_zero():
  w = jnp.array([0.0, 0.0, 0.0])
  R = exp(w)
  assert not jnp.any(jnp.isnan(R))
  assert jnp.allclose(R, jnp.eye(3), atol=1e-6)

def test_log_small_angle():
  w = jnp.array([1e-9, 1e-9, 1e-9])
  R = exp(w)
  w_recovered = log(R)
  assert not jnp.any(jnp.isnan(w_recovered))

def test_log_zero():
  R = jnp.eye(3)
  w = log(R)
  assert not jnp.any(jnp.isnan(w))
  assert jnp.allclose(w, jnp.zeros(3), atol=1e-6)

def test_Jl_identity():
  w = jnp.zeros(3)
  assert jnp.allclose(Jl(w), jnp.eye(3), atol=1e-6)

def test_Jr_identity():
  w = jnp.zeros(3)
  assert jnp.allclose(Jr(w), jnp.eye(3), atol=1e-6)

def test_Jl_inv_inverse():
  w = jnp.array([0.1, 0.2, 0.3])
  assert jnp.allclose(Jl(w) @ Jl_inv(w), jnp.eye(3), atol=1e-6)

def test_Jr_inv_inverse():
  w = jnp.array([0.1, 0.2, 0.3])
  assert jnp.allclose(Jr(w) @ Jr_inv(w), jnp.eye(3), atol=1e-6)

def test_Jr_is_Jl_negated():
  w = jnp.array([0.1, 0.2, 0.3])
  assert jnp.allclose(Jr(w), Jl(-w), atol=1e-6)

def test_Jl_small_angle():
  w = jnp.array([1e-9, 1e-9, 1e-9])
  assert not jnp.any(jnp.isnan(Jl(w)))

def test_Jr_small_angle():
  w = jnp.array([1e-9, 1e-9, 1e-9])
  assert not jnp.any(jnp.isnan(Jr(w)))

def test_Jl_zero():
  w = jnp.zeros(3)
  assert not jnp.any(jnp.isnan(Jl(w)))

def test_Jl_inv_zero():
  w = jnp.zeros(3)
  assert not jnp.any(jnp.isnan(Jl_inv(w)))