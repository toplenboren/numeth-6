import math
import matplotlib.pyplot as plt

e = math.e

THRESHOLD = 5e-05
BETA = (e + (1 / e) - 2)


def f(x, y):
    return y + 10 + 4 * x * (1 - x)


def q(x):
    return 10 + 4 * x * (1 - x)


def phi(x):
    return x - BETA


def get_points(N, a=0, b=1):
    h = (b - a) / N
    points = []
    cur_point = a
    for i in range(N + 1):
        points.append(cur_point)
        cur_point += h
    return points, h


def get_exact_solution(x):
    return (4 * x * x) - (4 * x) + (e ** (-x)) + (e ** x) - 2


def sweeping_algo(points, h, N):
    if not abs(2 + h ** 2) > 2:
        return

    result = [0.0] + [0.0 for _ in range(1, N)] + [BETA]
    l = [0] + [0 for _ in range(N)]
    mu = [0.0] + [0.0 for _ in range(1, N)] + [BETA]

    # Forward
    for i in range(1, N + 1):
        curr_l = l[i - 1]
        curr_mu = mu[i - 1]
        A = 2 + h ** 2
        B = (h ** 2) * q(points[i - 1])
        l[i] = 1 / (A - curr_l)
        mu[i] = (curr_mu - B) / (A - curr_l)

    # Backward
    for i in range(1, N):
        result[-i - 1] = l[-i] * result[-i] + mu[-i]

    return result


def shooting_algo(points, h):
    def runge_iter_x(x, y, h, cur_val):
        K1 = h * f(x, y)
        K2 = h * f(x + h / 2, y + K1 / 2)
        K3 = h * f(x + h / 2, y + K2 / 2)
        K4 = h * f(x + h, y + K3)
        return cur_val + (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)

    def runge_iter_y(x, cur_z, h, cur_y):
        K1 = h * cur_z
        K2 = h * runge_iter_x(x + h / 2, cur_y + K1 / 2, h, cur_z)
        K3 = h * runge_iter_x(x + h / 2, cur_y + K2 / 2, h, cur_z)
        K4 = h * runge_iter_x(x + h, cur_y + K3, h, cur_z)
        return cur_y + (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)

    def runge_kutta(points, h, n):
        result_z = [n]
        cur_z = n
        result_y = [0]
        cur_y = 0

        for x in points[1:]:
            cur_y = runge_iter_y(x, cur_z, h, cur_y)
            cur_z = runge_iter_x(x, cur_y, h, cur_z)

            result_z.append(cur_z)
            result_y.append(cur_y)

        return result_z, result_y

    mu_1 = -10
    mu_2 = 10

    def moving_chord_solve(a, b):

        f = lambda x: phi(runge_kutta(points, h, x)[1][-1])

        x = [a, b]
        n = 1

        while True:
            x.append(x[n] - f(x[n]) / (f(x[n]) - f(x[n - 1])) * (x[n] - x[n - 1]))
            if abs(x[n + 1] - x[n]) < THRESHOLD:
                break
            n += 1

        return x[-1]

    mu = moving_chord_solve(mu_1, mu_2)
    return runge_kutta(points, h, mu)[1]


def get_result(N):
    print()
    points, h = get_points(N)

    plt.title(f"Exact and calculated solutions for N={N}")
    plt.plot(points, list(map(lambda x: get_exact_solution(x), points)))
    plt.plot(points, shooting_algo(points, h))
    plt.plot(points, sweeping_algo(points, h, N))
    plt.legend(["Exact solution", "Shooting algorithm", "Sweeping algorithm"])
    plt.show()


if __name__ == '__main__':
    get_result(10)
    get_result(20)
