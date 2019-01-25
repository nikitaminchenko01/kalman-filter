import numpy as np
import math
import matplotlib.pyplot as plt
from visualizer import KalmanFilterVisualizer


def save_plots(real_position, measured_position, kalman_position, real_speed, kalman_speed, measured_error,
               kalman_error, times):
    plt.plot(times, real_position, label='Faktyczne położenie')
    plt.plot(times, measured_position, label='Zmierzone położenie')
    plt.plot(times, kalman_position, label='Estymowane położenie')
    plt.xlabel('czas (s)')
    plt.ylabel('położenie (m)')
    plt.legend()
    plt.title('Położenie')
    plt.savefig('positions.png')
    plt.clf()
    plt.plot(times[200:400], real_position[200:400], label='Faktyczne położenie')
    plt.plot(times[200:400], measured_position[200:400], label='Zmierzone położenie')
    plt.plot(times[200:400], kalman_position[200:400], label='Estymowane położenie')
    plt.legend()
    plt.xlabel('czas (s)')
    plt.ylabel('położenie (m)')
    plt.title('Położenie pomiędzy 2 a 4 sekundą ruchu')
    plt.savefig('positions24.png')
    plt.clf()

    plt.plot(times, real_speed, label='Faktyczna prędkość')
    plt.plot(times, kalman_speed, label='Estymowana prędkość')
    plt.xlabel('czas (s)')
    plt.ylabel('prędkość (m/s)')
    plt.legend()
    plt.title('Prędkość')
    plt.savefig('speed.png')
    plt.clf()
    plt.plot(times[200:400], real_speed[200:400], label='Faktyczna prędkość')
    plt.plot(times[200:400], kalman_speed[200:400], label='Estymowana prędkość')
    plt.legend()
    plt.xlabel('czas (s)')
    plt.ylabel('szybkość (m/s)')
    plt.title('Szybkość 2s-4s')
    plt.savefig('speed24.png')
    plt.clf()

    plt.plot(times, measured_error, label='Błąd odczytu położenia')
    plt.plot(times, kalman_error, label='Błąd estymacji położenia filtrem Kalmana')
    plt.xlabel('czas (s)')
    plt.ylabel('błąd (m)')
    plt.legend()
    plt.title('Przebieg błędu')
    plt.savefig('error.png')
    plt.clf()
    plt.plot(times[200:400], measured_error[200:400], label='Błąd odczytu położenia')
    plt.plot(times[200:400], kalman_error[200:400], label='Błąd estymacji położenia filtrem Kalmana')
    plt.legend()
    plt.xlabel('czas (s)')
    plt.ylabel('błąd (m)')
    plt.title('Przebieg błędu pomiędzy 2 a 4 sekundą ruchu')
    plt.savefig('error24.png')
    plt.clf()


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
        error = v_zadane - xhat[1]
        errorSum += (error * timestep)
        u = kp * error + ki * errorSum
        if(u > u_max):
            u = u_max
        errors[i + 1] = error
        u_val[i + 1] = u

        if i % iter_change == 0 and i != 0 :
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

    history, measurement_history, times, kalman, _, _ = simulate([10, 20, 30, 10], 100, 1, 0.1, 0)
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