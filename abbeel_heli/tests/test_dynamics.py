import numpy as np

from abbeel_heli.dynamics.powered_dynamics import PoweredHelicopterDynamics, PoweredDynamicsParams


def test_powered_dynamics_step_shape():
    dyn = PoweredHelicopterDynamics(PoweredDynamicsParams())
    state = dyn.hover_state(position_world=np.array([0.0, 0.0, 5.0]))
    action = dyn.hover_action()

    next_state = dyn.step(state, action)
    assert next_state.shape == state.shape
    assert dyn.state_dim == 13
    assert dyn.action_dim == 4


def test_powered_dynamics_small_step():
    dyn = PoweredHelicopterDynamics(PoweredDynamicsParams())
    state = dyn.hover_state(position_world=np.array([0.0, 0.0, 5.0]))
    action = dyn.hover_action()

    x = state.copy()
    for _ in range(100):
        x = dyn.step(x, action)

    assert np.all(np.isfinite(x))
