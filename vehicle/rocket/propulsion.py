from dataclasses import dataclass
from vehicle.base.component import RocketComponent
import jax.numpy as jnp

@dataclass
class PropulsionSystem(RocketComponent):
  thrust: jnp.ndarray
  burn_time: float
  exit_area: float
  exit_pressure: float

@dataclass
class SolidMotor(PropulsionSystem):
  length: float
  diameter: float
  mass_flow_rate: float