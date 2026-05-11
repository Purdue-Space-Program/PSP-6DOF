import jax.numpy as jnp
import matplotlib.pyplot as plt
from lie import se23
from integrator import mk_rk4, rk4

# constant rotation dynamics
def dynamics_mk(X, t):
  return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

def dynamics_rk4(X, t):
  xi = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
  return se23.hat(xi)

def run_sim(step_fn, dynamics, X0, dt, n_steps):
  X = X0
  states = [X]
  for i in range(n_steps):
    X = step_fn(dynamics, X, i * dt, dt)
    states.append(X)
  return states

def manifold_error(X):
  """How far R^T R is from identity."""
  R = X[:3, :3]
  return jnp.linalg.norm(R.T @ R - jnp.eye(3))

def bench_manifold_drift():
  X0 = jnp.eye(5)
  dt = 0.01
  n_steps = 1000

  mk_states = run_sim(mk_rk4.step, dynamics_mk, X0, dt, n_steps)
  rk4_states = run_sim(rk4.step, dynamics_rk4, X0, dt, n_steps)

  mk_errors = [manifold_error(X) for X in mk_states]
  rk4_errors = [manifold_error(X) for X in rk4_states]

  times = [i * dt for i in range(n_steps + 1)]

  plt.figure()
  plt.semilogy(times, mk_errors, label='MK-RK4')
  plt.semilogy(times, rk4_errors, label='Standard RK4')
  plt.xlabel('time (s)')
  plt.ylabel('manifold error ||R^T R - I||')
  plt.title('Manifold drift over time')
  plt.legend()
  plt.savefig('benchmarks/manifold_drift.png')
  plt.show()

def bench_accuracy_vs_stepsize():
  """Compare final state error vs stepsize — replicates Figure 3.1 from Munthe-Kaas."""
  X0 = jnp.eye(5)
  t_end = 1.0
  stepsizes = [0.5, 0.25, 0.1, 0.05, 0.01, 0.005]

  mk_errors = []
  rk4_errors = []

  # reference solution with very small stepsize
  dt_ref = 0.0001
  n_ref = int(t_end / dt_ref)
  ref_states = run_sim(mk_rk4.step, dynamics_mk, X0, dt_ref, n_ref)
  X_ref = ref_states[-1]

  for dt in stepsizes:
    n_steps = int(t_end / dt)
    mk_states = run_sim(mk_rk4.step, dynamics_mk, X0, dt, n_steps)
    rk4_states = run_sim(rk4.step, dynamics_rk4, X0, dt, n_steps)

    mk_errors.append(jnp.linalg.norm(mk_states[-1] - X_ref))
    rk4_errors.append(jnp.linalg.norm(rk4_states[-1] - X_ref))

  plt.figure()
  plt.loglog(stepsizes, mk_errors, 'x-', label='MK-RK4')
  plt.loglog(stepsizes, rk4_errors, '+-', label='Standard RK4')
  plt.xlabel('stepsize h')
  plt.ylabel('global error')
  plt.title('Accuracy vs stepsize (replicates Munthe-Kaas Fig 3.1)')
  plt.legend()
  plt.savefig('benchmarks/accuracy_vs_stepsize.png')
  plt.show()

if __name__ == '__main__':
  bench_manifold_drift()
  bench_accuracy_vs_stepsize()