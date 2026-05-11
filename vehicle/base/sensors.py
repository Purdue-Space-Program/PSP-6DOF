from dataclasses import dataclass
from vehicle.base.component import RocketComponent
import jax.numpy as jnp

@dataclass
class Sensor:
  name: str
  sampling_rate: float
  variance: float
  resolution: float
  bias: float
  scale_factor: float

@dataclass
class Altimeter(Sensor):
  pass

@dataclass
class GNSS(Sensor):
  pos_variance: jnp.ndarray
  vel_variance: jnp.ndarray

@dataclass
class Magnetometer(Sensor):
  pass

@dataclass
class Accelerometer(Sensor):
  bias_random_walk_rate: float = 6.86e-4

@dataclass
class Gyroscope(Sensor):
  bias_random_walk_rate: float = 1.3e-7