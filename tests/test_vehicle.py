import pytest
import jax.numpy as jnp
from vehicle.rocket.rocket import Rocket
from vehicle.rocket.propulsion import SolidMotor
from vehicle.rocket.structural import Tank, Fins

def test_rocket_instantiation():
  rocket = Rocket(
    name="test_rocket",
    length=3.0,
    outer_diameter=0.15,
    aero_data=jnp.zeros((300, 10))
  )
  assert rocket.name == "test_rocket"
  assert rocket.length == 3.0
  assert rocket.outer_diameter == 0.15

def test_reference_area():
  rocket = Rocket(
    name="test",
    length=2.0,
    outer_diameter=0.2,  # radius = 0.1
    aero_data=jnp.zeros((300, 10))
  )
  expected_area = jnp.pi * 0.1**2
  assert jnp.allclose(rocket.reference_area(), expected_area)

def test_total_mass_empty():
  rocket = Rocket(
    name="test",
    length=2.0,
    outer_diameter=0.2,
    aero_data=jnp.zeros((300, 10))
  )
  assert rocket.total_mass() == 0

def test_total_mass_with_components():
  tank = Tank(
    name="tank1",
    mass=5.0,
    position=jnp.array([0.5, 0.0, 0.0]),
    length=1.0,
    diameter=0.15,
    thickness=0.005,
    fuel_ox="Ox",
    liquid_mass=10.0
  )
  motor = SolidMotor(
    name="motor",
    mass=2.0,
    position=jnp.array([2.0, 0.0, 0.0]),
    thrust=jnp.array([5000.0]),
    burn_time=10.0,
    exit_area=0.01,
    exit_pressure=101325,
    length=0.5,
    diameter=0.1,
    mass_flow_rate=0.1
  )
  rocket = Rocket(
    name="test",
    length=3.0,
    outer_diameter=0.2,
    aero_data=jnp.zeros((300, 10)),
    components=[tank, motor]
  )
  assert jnp.isclose(rocket.total_mass(), 17.0)

def test_get_motor():
  motor = SolidMotor(
    name="motor",
    mass=2.0,
    position=jnp.array([2.0, 0.0, 0.0]),
    thrust=jnp.array([5000.0]),
    burn_time=10.0,
    exit_area=0.01,
    exit_pressure=101325,
    length=0.5,
    diameter=0.1,
    mass_flow_rate=0.1
  )
  rocket = Rocket(
    name="test",
    length=3.0,
    outer_diameter=0.2,
    aero_data=jnp.zeros((300, 10)),
    components=[motor]
  )
  retrieved = rocket.get_motor()
  assert retrieved is not None
  assert retrieved.name == "motor"

def test_get_motor_none():
  rocket = Rocket(
    name="test",
    length=3.0,
    outer_diameter=0.2,
    aero_data=jnp.zeros((300, 10))
  )
  assert rocket.get_motor() is None