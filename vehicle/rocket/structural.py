from dataclasses import dataclass
from vehicle.base.component import RocketComponent
from typing import Literal

@dataclass
class PointMass(RocketComponent):
  pass

@dataclass 
class Fins(RocketComponent):
  count: int
  span: float
  root_chord: float
  tip_chord: float
  sweep: float
  thickness: float

@dataclass
class Tank(RocketComponent):
  length: float
  diameter: float
  thickness: float
  fuel_ox: Literal['Fuel', 'Ox']
  liquid_mass: float