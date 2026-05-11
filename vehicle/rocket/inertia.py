import jax.numpy as jnp
from vehicle.rocket.rocket import Rocket
from vehicle.rocket.propulsion import PropulsionSystem, SolidMotor
from vehicle.rocket.structural import Tank, Fins

def variable_com_moi(rocket: Rocket) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  motor = rocket.get_motor()
  if motor is None:
     raise ValueError("Rocket must have a motor")
  burn_time = motor.burn_time
  N = int(burn_time * 100)
  times = jnp.linspace(0, burn_time, N)

  length_tot = rocket.length
  radius_tot = rocket.outer_diameter / 2
  rocket_mass = rocket.total_mass()

  # running CoM and counted mass
  com_x = jnp.zeros(N)
  com_y = jnp.zeros(N)
  com_z = jnp.zeros(N)
  counted_mass = jnp.zeros(N)

  moi_x = jnp.zeros(N)
  moi_y = jnp.zeros(N)
  moi_z = jnp.zeros(N)

  # ---- CoM pass ----
  for comp in rocket.components:
      if isinstance(comp, Tank):
          mass = comp.mass
          mass_liquid = comp.liquid_mass
          length = comp.length
          radius = comp.diameter / 2
          thick = comp.thickness
          pos = comp.position

          drain_rate_height = length / burn_time
          drain_rate_mass = mass_liquid / burn_time
          liquid_level = length - drain_rate_height * times
          liquid_mass = mass_liquid - drain_rate_mass * times
          liquid_locate = length - liquid_level

          tank_x = (pos[0] + length / 2) * jnp.ones(N)
          tank_y = pos[1] * jnp.ones(N)
          tank_z = pos[2] * jnp.ones(N)

          liquid_x = pos[0] + liquid_locate + liquid_level / 2
          liquid_y = tank_y
          liquid_z = tank_z

          total_tank_mass = liquid_mass + mass
          tank_tot_x = (tank_x * mass + liquid_x * liquid_mass) / total_tank_mass
          tank_tot_y = (tank_y * mass + liquid_y * liquid_mass) / total_tank_mass
          tank_tot_z = (tank_z * mass + liquid_z * liquid_mass) / total_tank_mass

          com_x = (com_x * counted_mass + tank_tot_x * total_tank_mass) / (counted_mass + total_tank_mass)
          com_y = (com_y * counted_mass + tank_tot_y * total_tank_mass) / (counted_mass + total_tank_mass)
          com_z = (com_z * counted_mass + tank_tot_z * total_tank_mass) / (counted_mass + total_tank_mass)
          counted_mass = counted_mass + total_tank_mass

      elif isinstance(comp, SolidMotor):
          mass = comp.mass
          length = comp.length
          pos = comp.position

          engine_length_rate = length / burn_time
          mass_loss_rate = mass / burn_time
          propellant_length = length - engine_length_rate * times
          propellant_mass = mass - mass_loss_rate * times
          propellant_locate = length - propellant_length

          motor_x = pos[0] + propellant_locate + propellant_length / 2
          motor_y = pos[1] * jnp.ones(N)
          motor_z = pos[2] * jnp.ones(N)

          com_x = (com_x * counted_mass + motor_x * propellant_mass) / (counted_mass + propellant_mass)
          com_y = (com_y * counted_mass + motor_y * propellant_mass) / (counted_mass + propellant_mass)
          com_z = (com_z * counted_mass + motor_z * propellant_mass) / (counted_mass + propellant_mass)
          counted_mass = counted_mass + propellant_mass

      elif isinstance(comp, Fins):
          fin_mass = comp.mass * jnp.ones(N)
          counted_mass = counted_mass + fin_mass

  # structural residual
  structure_mass = rocket_mass - counted_mass[0]
  structure_mass = jnp.maximum(structure_mass, 0.0)
  com_struct_x = (length_tot / 2) * jnp.ones(N)

  com_x = (com_x * counted_mass + com_struct_x * structure_mass) / (structure_mass + counted_mass)
  com_y = (com_y * counted_mass + 0.0 * structure_mass) / (structure_mass + counted_mass)
  com_z = (com_z * counted_mass + 0.0 * structure_mass) / (structure_mass + counted_mass)

  tot_mass = jnp.stack([times, structure_mass + counted_mass], axis=1)
  com = jnp.stack([times, com_x, com_y, com_z], axis=1)

  # ---- MoI pass ----
  for comp in rocket.components:
    if isinstance(comp, Tank):
      mass = comp.mass
      mass_liquid = comp.liquid_mass
      length = comp.length
      radius = comp.diameter / 2
      thick = comp.thickness
      pos = comp.position

      drain_rate_height = length / burn_time
      drain_rate_mass = mass_liquid / burn_time
      liquid_level = length - drain_rate_height * times
      liquid_mass = mass_liquid - drain_rate_mass * times
      liquid_locate = length - liquid_level

      tank_x = (pos[0] + length / 2) * jnp.ones(N)
      liquid_x = pos[0] + liquid_locate + liquid_level / 2

      # dry tank hollow cylinder
      tank_moi_x = 0.5 * mass * (radius**2 + (radius - thick)**2)
      tank_moi_y = (1/12) * mass * (3 * (radius**2 + (radius - thick)**2) + length**2)
      tank_moi_z = tank_moi_y

      moi_x = moi_x + tank_moi_x + mass * (com_y**2 + com_z**2)
      moi_y = moi_y + tank_moi_y + mass * (jnp.abs(tank_x - com_x)**2 + com_z**2)
      moi_z = moi_z + tank_moi_z + mass * (jnp.abs(tank_x - com_x)**2 + com_y**2)

      # liquid solid cylinder
      liquid_moi_x = 0.5 * liquid_mass * (radius - thick)**2
      liquid_moi_y = (1/12) * liquid_mass * (3 * (radius - thick)**2 + liquid_level**2)
      liquid_moi_z = liquid_moi_y

      moi_x = moi_x + liquid_moi_x + liquid_mass * (com_y**2 + com_z**2)
      moi_y = moi_y + liquid_moi_y + liquid_mass * (jnp.abs(liquid_x - com_x)**2 + com_z**2)
      moi_z = moi_z + liquid_moi_z + liquid_mass * (jnp.abs(liquid_x - com_x)**2 + com_y**2)

    elif isinstance(comp, SolidMotor):
      mass = comp.mass
      length = comp.length
      pos = comp.position

      engine_length_rate = length / burn_time
      mass_loss_rate = mass / burn_time
      propellant_length = length - engine_length_rate * times
      propellant_mass = mass - mass_loss_rate * times
      propellant_locate = length - propellant_length
      propellant_x = pos[0] + propellant_locate + propellant_length / 2

      motor_moi_x = 0.5 * propellant_mass * radius_tot**2
      motor_moi_y = (1/12) * propellant_mass * (3 * radius_tot**2 + propellant_length**2)
      motor_moi_z = motor_moi_y

      moi_x = moi_x + motor_moi_x + propellant_mass * (com_y**2 + com_z**2)
      moi_y = moi_y + motor_moi_y + propellant_mass * (jnp.abs(com_x - propellant_x)**2 + com_z**2)
      moi_z = moi_z + motor_moi_z + propellant_mass * (jnp.abs(com_x - propellant_x)**2 + com_y**2)

  # structure MoI
  struct_moi_x = 0.5 * structure_mass * radius_tot**2
  struct_moi_y = (1/12) * structure_mass * (3 * radius_tot**2 + length_tot**2)
  struct_moi_z = struct_moi_y

  moi_x = moi_x + struct_moi_x + structure_mass * (com_y**2 + com_z**2)
  moi_y = moi_y + struct_moi_y + structure_mass * (jnp.abs(length_tot/2 - com_x)**2 + com_z**2)
  moi_z = moi_z + struct_moi_z + structure_mass * (jnp.abs(length_tot/2 - com_x)**2 + com_y**2)

  moi = jnp.stack([times, moi_x, moi_y, moi_z], axis=1)

  return tot_mass, com, moi


def interpolate_at_time(table: jnp.ndarray, t: float) -> jnp.ndarray:
  times = table[:, 0]
  idx = jnp.argmin(jnp.abs(times - t))
  return table[idx, 1:]