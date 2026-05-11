import pytest
import jax.numpy as jnp
from dynamics.forces import gravity_mag, gravity

def test_gravity_mag_low_fidelity():
  g_sea_level = gravity_mag(0.0, 0.0, 'low')
  g_high_alt = gravity_mag(10000.0, 45.0, 'low')
  assert jnp.isclose(g_sea_level, 9.8)
  assert jnp.isclose(g_high_alt, 9.8)

def test_gravity_mag_high_fidelity_sea_level():
  g = gravity_mag(0.0, 0.0, 'high')
  assert 9.78 < g < 9.79  # ~9.7803 at equator

def test_gravity_mag_high_fidelity_pole():
  g = gravity_mag(0.0, 90.0, 'high')
  assert 9.83 < g < 9.84  # ~9.8322 at pole

def test_gravity_mag_decreases_with_altitude():
  g_sea = gravity_mag(0.0, 45.0, 'high')
  g_10km = gravity_mag(10000.0, 45.0, 'high')
  g_100km = gravity_mag(100000.0, 45.0, 'high')
  assert g_sea > g_10km > g_100km

def test_gravity_force_direction():
  mass = 100.0  # kg
  g = jnp.array(9.8)
  f_grav = gravity(mass, g)
  expected = jnp.array([0.0, 0.0, -980.0])  # -mass * g in Z
  assert jnp.allclose(f_grav, expected)

def test_gravity_force_magnitude():
  mass = 50.0
  g = jnp.array(9.81)
  f_grav = gravity(mass, g)
  magnitude = jnp.linalg.norm(f_grav)
  expected_mag = mass * g
  assert jnp.isclose(magnitude, expected_mag)

def test_gravity_force_zero_mass():
  f_grav = gravity(0.0, jnp.array(9.8))
  assert jnp.allclose(f_grav, jnp.zeros(3))