from dataclasses import dataclass, field
import jax.numpy as jnp
from vehicle.base.component import RocketComponent
from vehicle.rocket.structural import Tank

@dataclass
class Rocket:
  name: str
  length: float
  outer_diameter: float
  aero_data: jnp.ndarray
  components: list[RocketComponent] = field(default_factory=list)

  def reference_area(self) -> float:
    import jax.numpy as jnp
    r = self.outer_diameter / 2
    return jnp.pi * r**2

  def total_mass(self) -> float:
    total = 0
    for c in self.components:
      total += c.mass
      if isinstance(c, Tank):
        total += c.liquid_mass
    return total

  def get_motor(self):
    from vehicle.rocket.propulsion import PropulsionSystem
    for c in self.components:
      if isinstance(c, PropulsionSystem):
        return c
    return None