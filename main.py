import numpy as np
import math
import matplotlib.pyplot as plt
from visualizer import KalmanFilterVisualizer


def save_plots(real_position, measured_position, kalman_position, real_speed, kalman_speed, measured_error,
               kalman_error, timesteps):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    times = np.cumsum(timesteps)
    ax1.plot(times, real_position, label='Истинная координата')
    ax1.plot(times, measured_position, label='Измеренная координата')
    ax1.plot(times, kalman_position, label='Оцененная координата')
    ax1.set_xlabel('время')
    ax1.set_ylabel('координата')
    ax1.legend()
    ax1.set_title('График положения')

    ax2 = fig.add_subplot(222)
    ax2.plot(times, real_speed, label='Фактическая скорость')
    ax2.plot(times, kalman_speed, label='Оцененная скорость')
    ax2.set_xlabel('время')
    ax2.set_ylabel('скорость (м/с)')
    ax2.legend()
    ax2.set_title('Скорость')

    ax3 = fig.add_subplot(223)
    ax3.plot(times, measured_error, label='Ошибка измерения координаты')
    ax3.plot(times, kalman_error, label='Ошибка оценки координаты фильтром Калмана')
    ax3.set_xlabel('время')
    ax3.set_ylabel('ошибка')
    ax3.legend()
    ax3.set_title('Ошибка координаты')
    fig.show()

    ax4 = fig.add_subplot(224)
    ax4.plot(np.arange(0, 10, 0.01), timesteps, label='Временной шаг')
    #ax4.plot(times, kalman_error, label='Ошибка оценки координаты фильтром Калмана')
    ax4.set_xlabel('номер измерения')
    ax4.set_ylabel('временной шаг')
    ax4.legend()
    ax4.set_title('Временной шаг')
    fig.show()



def num_iters(T, h):
    return int(math.ceil(T/h))


def simulate(velocity_list, T, kp, ki, kd):
    timestep = 0.01
    iters = num_iters(T, timestep)
    iter_change = round(iters / len(velocity_list))
    last_velocity_index = 0

    def timestep_dynamic(i):
        return 0.01 + 0.01 * np.sin(0.01 * i)

    times = [timestep]
    u_max = 3
    accelnoise = 0.06
    x0 = np.array([[0], [0]])
    A = np.array([[1, timestep], [0, 1]])
    B = np.array([[1/2 * (timestep ** 2)], [timestep]])
    C = np.array([[1, 0]])
    measnoise = 3

    v_known = velocity_list[last_velocity_index]


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
        error = v_known - xhat[1]
        timestep = timestep_dynamic(i)
        times.append(timestep)
        A = np.array([[1, timestep], [0, 1]])
        B = np.array([[1 / 2 * (timestep ** 2)], [timestep]])
        C = np.array([[1, 0]])
        Sw = accelnoise ** 2 * np.array([[timestep ** 4 / 4, timestep ** 3 / 2], [timestep ** 3 / 2, timestep ** 2]])
        #P = Sw
        errorSum += (error * timestep)
        u = kp * error + ki * errorSum
        if(u > u_max):
            u = u_max
        errors[i + 1] = error
        u_val[i + 1] = u

        if i % iter_change == 0 and i != 0 :
            last_velocity_index += 1
            v_known = velocity_list[last_velocity_index]

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
    #timestep = 0.01
    return state_history, measurement_history, times, kalman_history, errors, u_val # np.arange(0, T, timestep)


def test_statistically():
    errors_test = np.zeros((1000, 10000))
    errors_test_kalman = np.zeros((1000, 10000))
    for i in range(1000):
        print(i)
        history, measurement_history, times, kalman, _, _ = simulate([10, 20, 30, 10], 100, 1, 0.1, 0)
        errors = measurement_history - np.squeeze(history[:, 0, :])
        error_after = np.squeeze(kalman[:, 0, :]) - np.squeeze(history[:, 0, :])
        errors_test[i] = errors
        errors_test_kalman[i] = error_after

    errors_avg = np.average(np.abs(errors_test))
    errors_kalman_avg = np.average(np.abs(errors_test_kalman))

    errors_std_dev = np.std(errors_test)
    errors_kalman_std_dev = np.std(errors_test_kalman)
    return errors_avg, errors_std_dev, errors_kalman_avg, errors_kalman_std_dev


if __name__ == '__main__':
    # test_statistically()

    history, measurement_history, times, kalman, _, _ = simulate([10, 20, 30, 10], 10, 1, 0.1, 0)
    errors = measurement_history - np.squeeze(history[:, 0, :])
    error_after = np.squeeze(kalman[:, 0, :]) - np.squeeze(history[:, 0, :])

    save_plots(history[:, 0, :], measurement_history, kalman[:, 0, :], history[:, 1, :], kalman[:, 1, :],
               errors, error_after, times)


    kalman_errors = error_after
    measured_errors = errors
    velocities = np.squeeze(history[:, 1, :])
    kalman_velocities = np.squeeze(kalman[:, 1, :])
    visualizer = KalmanFilterVisualizer(measured_errors, kalman_errors, velocities, kalman_velocities, 100)
    visualizer.run()