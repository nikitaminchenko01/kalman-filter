import numpy as np
import math
import matplotlib.pyplot as plt
from game import run
from scipy.optimize import minimize
from functools import reduce
import threading

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


def simulate(velocity_list, T, kp, ki, kd):
    timestep = 0.01
    iters = num_iters(T, timestep)
    iter_change = round(iters / len(velocity_list))
    last_velocity_index = 0

    u_max = 4
    accelnoise = 0.06
    x0 = np.array([[0], [0]])
    A = np.array([[1, timestep], [0, 1]])
    B = np.array([[1/2 * (timestep ** 2)], [timestep]])
    C = np.array([[1, 0]])
    measnoise = 3

    v_zadane = velocity_list[last_velocity_index]

    Sz = measnoise ** 2
    Sw = accelnoise ** 2 * np.array([[timestep ** 4 / 4, timestep ** 3 / 2], [timestep ** 3 / 2, timestep ** 2]])
    P = Sw

    errors = np.zeros(iters)
    errorSum = errors[0] * timestep
    u_val = np.zeros(iters)

    xhat = x0
    state_history = np.zeros((iters, 2, 1))
    kalman_history = np.zeros((iters, 2, 1))
    state_history[0] = x0
    kalman_history[0] = xhat
    measurement_history = np.zeros(iters)
    for i in range(iters - 1):
        # u = 3 * np.sin(timestep * i) + 2
        # print(u)
        error = v_zadane - xhat[1]
        errorSum += (error * timestep)
        u = kp * error + ki * errorSum
        if(u > u_max):
            u = u_max
        errors[i + 1] = error
        u_val[i + 1] = u

        if(i % iter_change == 0 and i != 0):
            last_velocity_index += 1
            v_zadane = velocity_list[last_velocity_index]

        process_noise = accelnoise * np.array([[1/2 * (timestep**2) * np.random.normal()], [timestep * np.random.normal()]])
        x = A@(state_history[i]) + B*u + process_noise
        measure_noise = np.random.normal() * measnoise
        y = C @ x + measure_noise

        xhat = A @ xhat + B * u
        inn = y - C @ xhat
        s = C @ P @ C.T + Sz
        K = A @ P @ C.T @ np.linalg.inv(s)
        xhat = xhat + K @ inn
        P = A @ P @ A.T - A @ P @ C.T @ np.linalg.inv(s) @ C @ P @ A.T + Sw
        kalman_history[i+1] = xhat
        state_history[i+1] = x
        measurement_history[i+1] = y


    return state_history, measurement_history, np.arange(0, T, timestep), kalman_history, errors, u_val


def to_minimize(x):
    kp, ki, kd = x[0], x[1], x[2]
    _, _, _, _, _, e = simulate(kp, ki, kd)
    return reduce(lambda acc, err: acc + err**2, e, 0)

if __name__ == '__main__':
    history, measurement_history, times, kalman, _, _ = simulate([10, 20, 30, 10], 100, 1, 0.1, 0)
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

    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.001)
    kalman_errors = error_after
    measured_errors = errors
    velocities = np.squeeze(history[:, 1, :])
    kalman_velocities = np.squeeze(kalman[:, 1, :])
    run(kalman_errors, measured_errors, velocities, kalman_velocities)