import numpy as np
import math
import matplotlib.pyplot as plt

def plot_n_vars(values, times):
    for name, value in values:
        plt.plot(times, value, label=name)
    plt.xlabel('t')
    plt.ylabel('stan')
    plt.legend()
    plt.show()


def plot_n_vars_subplots(values, times):
    n = len(values)
    cols = 1
    rows = int(math.ceil(n / cols))
    f, axes = plt.subplots(rows, cols)
    print(axes)
    for i in range(n):
        name, value = values[i]
        axes[i].plot(times, value)
        axes[i].set_xlabel('t')
        axes[i].set_ylabel(name)
    plt.show()



def num_iters(T, h):
    return int(math.ceil(T/h))


def simulate(T):
    u = 1
    a = 1
    accelnoise = 0.2
    timestep = 0.1
    iters = num_iters(T, timestep)
    x0 = np.array([[0], [0]])
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[1/2 * a * timestep ** 2], [0.1]])
    C = np.array([[1, 0]])
    measnoise = 10

    Sz = measnoise ** 2
    Sw = accelnoise ** 2 * np.array([[timestep ** 4 / 4, timestep ** 3 / 2], [timestep ** 3 / 2, timestep ** 2]])
    P = Sw

    xhat = x0
    state_history = np.zeros((iters, 2, 1))
    kalman_history = np.zeros((iters, 2, 1))
    state_history[0] = x0
    kalman_history[0] = xhat
    measurement_history = np.zeros(iters)
    for i in range(iters - 1):
        # u = 3 * np.sin(timestep * i) + 2
        # print(u)
        process_noise = accelnoise * np.array([[1/2 * timestep**2 * np.random.normal()], [timestep * np.random.normal()]])
        measure_noise = np.random.normal() * measnoise
        x = A@(state_history[i]) + B*u + process_noise
        y = C @ x + measure_noise

        xhat = A @ xhat + B * u
        inn = y - C @ xhat
        s = C @ P @ C.T + Sz
        K = A @ P @ C.T @ np.linalg.inv(s)
        xhat = xhat + K @ inn
        kalman_history[i+1] = xhat
        state_history[i+1] = x
        measurement_history[i+1] = y
    return state_history, measurement_history, np.arange(0, T, timestep), kalman_history


if __name__ == '__main__':
    history, measurement_history, times, kalman = simulate(20)
    errors = measurement_history - np.squeeze(history[:, 0, :])
    error_after = np.squeeze(kalman[:, 0, :]) - np.squeeze(history[:, 0, :])

    cols = 1
    rows = 3
    f, axes = plt.subplots(rows, cols)

    name, value = ['p', history[:, 0, :]]
    axes[0].plot(times, value, label=name)
    name, value = ['measured_p', measurement_history]
    axes[0].plot(times, value, label=name)
    name, value = ['kalman_p', kalman[:, 0, :]]
    axes[0].plot(times, value, label=name)

    name, value = ['v', history[:, 1, :]]
    axes[1].plot(times, value, label=name)
    name, value = ['kalman_v', kalman[:, 1, :]]
    axes[1].plot(times, value, label=name)

    name, value = ['errors', errors]
    axes[2].plot(times, value, label=name)

    name, value = ['errors_kalman', error_after]
    axes[2].plot(times, value, label=name)

    axes[0].set_xlabel('t')
    axes[0].set_ylabel('stan')
    axes[0].legend()

    axes[1].set_xlabel('t')
    axes[1].set_ylabel('predkosc')
    axes[1].legend()

    axes[2].set_xlabel('t')
    axes[2].set_ylabel('error')
    axes[2].legend()

    print(np.sum(error_after))
    plt.show()


