from dataclasses import dataclass, field
import jax.numpy as jnp

@dataclass
class RocketComponent:
  name: str
  mass: float
  position: jnp.ndarray