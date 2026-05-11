import jax.numpy as jnp
from lie import se23

def gravity_mag(alt: float, lat: float, fidelity: str) -> jnp.ndarray:
  if fidelity == 'low':
    return jnp.array(9.8)
  else:
    lat_rad = jnp.deg2rad(lat)
    Re = 6378137.0
    f = 1 / 298.257223563
    m = 0.00344978650684

    g0 = 9.7803253359 * (1 + 0.00193185265241 * jnp.sin(lat_rad)**2) / jnp.sqrt(1 - 0.00669437999014 * jnp.sin(lat_rad)**2)

    g = g0 * (1 - (2/Re) * (1 + f + m - 2*f*jnp.sin(lat_rad)**2) * alt + (3/Re**2) * alt**2)

    return g

def gravity(mass: float, g: jnp.ndarray) -> jnp.ndarray:
  return mass * g * jnp.array([0.0, 0.0, -1.0])