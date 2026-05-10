from lie import so3
import jax.numpy as jnp

def hat(xi: jnp.ndarray) -> jnp.ndarray: # R^9 -> se2(3), wedge operator
  p, v, w = xi[:3], xi[3:6], xi[6:]
  W = so3.hat(w)
  return jnp.block([
    [W, v.reshape(3,1), p.reshape(3,1)],
    [jnp.zeros((1,3)), jnp.zeros((1,1)), jnp.zeros((1,1))],
    [jnp.zeros((1,3)), jnp.zeros((1,1)), jnp.zeros((1,1))]
  ])

def vee(X: jnp.ndarray) -> jnp.ndarray: # se2(3) -> R^9
  p = X[:3, 4] # pos
  v = X[:3, 3] # vel
  w = so3.vee(X[:3, :3]) # rotation
  return jnp.concatenate([p, v, w])


def exp(xi: jnp.ndarray) -> jnp.ndarray: # se2(3) -> SE2(3)
  p, v, w = xi[:3], xi[3:6], xi[6:]
  R = so3.exp(w)
  Jl_so3 = so3.Jl(w)
  return jnp.block([
    [R, (Jl_so3 @ v).reshape(3,1), (Jl_so3 @ p).reshape(3,1)],
    [jnp.zeros((1,3)), jnp.ones((1,1)), jnp.zeros((1,1))],
    [jnp.zeros((1,3)), jnp.zeros((1,1)), jnp.ones((1,1))]
  ])


def log(X: jnp.ndarray) -> jnp.ndarray: # SE2(3) -> se2(3)
  R = X[:3, :3]
  v = X[:3, 3]
  p = X[:3, 4]
  w = so3.log(R)
  Jl_so3_inv = so3.Jl_inv(w)
  return jnp.concatenate([Jl_so3_inv @ p, Jl_so3_inv @ v, w])


def adjoint(X: jnp.ndarray) -> jnp.ndarray: # Ad for SE2(3)
  R = X[:3, :3]
  v = X[:3, 3]
  p = X[:3, 4]
  return jnp.block([
    [R, so3.hat(p) @ R, jnp.zeros((3,3))],
    [jnp.zeros((3,3)), R, so3.hat(v) @ R],
    [jnp.zeros((3,3)), jnp.zeros((3,3)), R]
  ])

def Qr(w: jnp.ndarray) -> jnp.ndarray: # equation 27
  angle = jnp.linalg.norm(w)
  W = so3.hat(w)
  W2 = W @ W
  safe_angle = jnp.where(angle < 1e-7, 1.0, angle)
  q0 = 0.5
  q1 = jnp.where(angle < 1e-7,
    1/6.0 - angle**2/120.0,
    (jnp.sin(safe_angle) - safe_angle*jnp.cos(safe_angle)) / safe_angle**3)
  q2 = jnp.where(angle < 1e-7,
    1/12.0 - angle**2/720.0,
    0.5/safe_angle**2 - jnp.sin(safe_angle)/safe_angle**3 - (jnp.cos(safe_angle) - 1)/safe_angle**4)
  return q0*jnp.eye(3) + q1*W + q2*W2


def Ql(w: jnp.ndarray) -> jnp.ndarray: # equation 28
  angle = jnp.linalg.norm(w)
  W = so3.hat(w)
  W2 = W @ W
  safe_angle = jnp.where(angle < 1e-7, 1.0, angle)
  coeff1 = jnp.where(angle < 1e-7, 0.5, (1 - jnp.cos(safe_angle)) / safe_angle**2)
  coeff2 = jnp.where(angle < 1e-7,
    (safe_angle - jnp.sin(safe_angle)) / safe_angle**3,
    (angle - jnp.sin(safe_angle)) / safe_angle**3)
  Jl_so3 = jnp.eye(3) + coeff1*W + coeff2*W2
  return Jl_so3 - Qr(w)


def Jr_inv(xi: jnp.ndarray) -> jnp.ndarray: # equation 33
  return Jl_inv(-xi)

def Jl_inv(xi: jnp.ndarray) -> jnp.ndarray: # equation 32
  p, v, w = xi[:3], xi[3:6], xi[6:]
  angle = jnp.linalg.norm(w)
  W = so3.hat(w)
  W2 = W @ W
  safe_angle = jnp.where(angle < 1e-7, 1.0, angle)
  # Sl = (Jl_SO3)^-1, equation 23
  coeff1 = 0.5
  coeff2 = jnp.where(angle < 1e-7,
      1/12.0 - angle**2/720.0,
      1/safe_angle**2 - (1 + jnp.cos(safe_angle))/(2*safe_angle*jnp.sin(safe_angle)))
  Sl = jnp.eye(3) - coeff1*W + coeff2*W2
  Ql_mat = Ql(w)
  c1 = Sl
  c2 = -Sl @ Ql_mat @ Sl
  c3 = -Sl @ (so3.hat(Ql_mat @ p)) @ Sl  # Ql(w;p)Sl
  c4 = -Sl @ (so3.hat(Ql_mat @ v)) @ Sl  # Ql(w;v)Sl
  return jnp.block([
      [c1, c2, c3],
      [jnp.zeros((3,3)), c1, c4],
      [jnp.zeros((3,3)), jnp.zeros((3,3)), c1]
  ])