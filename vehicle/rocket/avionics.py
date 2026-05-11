from vehicle.base.sensors import Altimeter, GNSS, Magnetometer, Accelerometer, Gyroscope
from dataclasses import dataclass, field
from typing import Optional
from vehicle.base.component import RocketComponent

@dataclass
class Avionics(RocketComponent):
  altimeter: Optional[Altimeter] = None
  gnss: Optional[GNSS] = None
  magnetometer: Optional[Magnetometer] = None
  accelerometer: Optional[Accelerometer] = None
  gyroscope: Optional[Gyroscope] = None