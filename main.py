import numpy as np
import math
import matplotlib.pyplot as plt
from game import run


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


if __name__ == '__main__':
    # errors_test = np.zeros((1000, 10000))
    # errors_test_kalman = np.zeros((1000, 10000))
    # for i in range(1000):
    #     print(i)
    #     history, measurement_history, times, kalman, _, _ = simulate([10, 20, 30, 10], 100, 1, 0.1, 0)
    #     errors = measurement_history - np.squeeze(history[:, 0, :])
    #     error_after = np.squeeze(kalman[:, 0, :]) - np.squeeze(history[:, 0, :])
    #     errors_test[i] = errors
    #     errors_test_kalman[i] = error_after
    #
    # errors_avg = np.average(np.abs(errors_test))
    # errors_kalman_avg = np.average(np.abs(errors_test_kalman))
    #
    # errors_std_dev = np.std(errors_test)
    # errors_kalman_std_dev = np.std(errors_test_kalman)
    #
    # print("AVG ERROR:")
    # print(errors_avg)
    #
    # print("AVG ERROR KALMAN:")
    # print(errors_kalman_avg)
    #
    # print("STD DEV ERROR:")
    # print(errors_std_dev)
    #
    # print("STD DEV ERROR KALMAN:")
    # print(errors_kalman_std_dev)


    history, measurement_history, times, kalman, _, _ = simulate([10, 20, 30, 10], 100, 1, 0.1, 0)
    errors = measurement_history - np.squeeze(history[:, 0, :])
    error_after = np.squeeze(kalman[:, 0, :]) - np.squeeze(history[:, 0, :])


    plt.plot(times, history[:, 0, :], label='Faktyczne położenie')
    plt.plot(times, measurement_history, label='Zmierzone położenie')
    plt.plot(times, kalman[:, 0, :], label='Odfiltrowane położenie')
    plt.xlabel('czas (s)')
    plt.ylabel('położenie (m)')
    plt.legend()
    plt.title('Położenie')
    plt.savefig('positions.png')
    plt.clf()
    plt.plot(times[200:400], history[200:400, 0, :], label='Faktyczne położenie')
    plt.plot(times[200:400], measurement_history[200:400], label='Zmierzone położenie')
    plt.plot(times[200:400], kalman[200:400, 0, :], label='Odfiltrowane położenie')
    plt.legend()
    plt.xlabel('czas (s)')
    plt.ylabel('położenie (m)')
    plt.title('Położenie 2s-4s')
    plt.savefig('positions24.png')
    plt.show()

    plt.plot(times, history[:, 1, :], label='Faktyczna szybkość')
    plt.plot(times, kalman[:, 1, :], label='Estymowana szybkosć')
    plt.xlabel('czas (s)')
    plt.ylabel('szybkość (m/s)')
    plt.legend()
    plt.title('Szybkość')
    plt.savefig('speed.png')
    plt.clf()
    plt.plot(times[200:400], history[200:400, 1, :], label='Faktyczna szybkość')
    plt.plot(times[200:400], kalman[200:400, 1, :], label='Estymowana szybkość')
    plt.legend()
    plt.xlabel('czas (s)')
    plt.ylabel('szybkość (m/s)')
    plt.title('Szybkość 2s-4s')
    plt.savefig('speed24.png')
    plt.show()

    plt.plot(times, errors, label='Błąd odczytu położenia')
    plt.plot(times, error_after, label='Błąd estymacji położenia filtrem Kalmana')
    plt.xlabel('czas (s)')
    plt.ylabel('błąd (m)')
    plt.legend()
    plt.title('Przebieg błędu')
    plt.savefig('error.png')
    plt.clf()
    plt.plot(times[200:400], errors[200:400], label='Błąd odczytu położenia')
    plt.plot(times[200:400], error_after[200:400], label='Błąd estymacji położenia filtrem Kalmana')
    plt.legend()
    plt.xlabel('czas (s)')
    plt.ylabel('błąd (m)')
    plt.title('Przebieg błędu pomiędzy 2 a 4 sekundą ruchu')
    plt.savefig('error24.png')

    kalman_errors = error_after
    measured_errors = errors
    velocities = np.squeeze(history[:, 1, :])
    kalman_velocities = np.squeeze(kalman[:, 1, :])
    run(kalman_errors, measured_errors, velocities, kalman_velocities)