# pylint: disable=invalid-name
# pylint: disable=missing-module-docstring, missing-function-docstring

import numpy as np
import scipy.optimize as opt

def calc_prev_p_matrix(curr_state, curr_p_mat, q_mat, r_mat, a_mat, b_mat):
    prev_state, prev_control, slack = one_step_optimization(curr_state, curr_p_mat,
                                                            q_mat, r_mat, a_mat,
                                                            b_mat)
    prev_p_mat = calc_p_from_opt(prev_state, prev_control, slack)

    return prev_p_mat, prev_state, prev_control

def one_step_optimization(curr_state, curr_p_mat, q_mat, r_mat, a_mat, b_mat):
    n = len(curr_state)
    m = b_mat.shape[1]
    l = curr_p_mat.shape[0]

    e_x_c_shape = (l, 1)
    e_x_p_shape = (n, 1)
    e_u_p_shape = (m, 1)

    A_ub = np.block([[curr_p_mat @ a_mat, curr_p_mat @ b_mat,
                      -np.ones(e_x_c_shape), np.zeros(e_x_c_shape),
                      np.zeros(e_x_c_shape)],
                     [-curr_p_mat @ a_mat, -curr_p_mat @ b_mat,
                      -np.ones(e_x_c_shape), np.zeros(e_x_c_shape),
                      np.zeros(e_x_c_shape)],
                     [q_mat, np.zeros((n, m)), np.zeros(e_x_p_shape),
                      -np.ones(e_x_p_shape), np.zeros(e_x_p_shape)],
                     [-q_mat, np.zeros((n, m)), np.zeros(e_x_p_shape),
                      -np.ones(e_x_p_shape), np.zeros(e_x_p_shape)],
                     [np.zeros((m, n)), r_mat, np.zeros(e_u_p_shape),
                      np.zeros(e_u_p_shape), -np.ones(e_u_p_shape)],
                     [np.zeros((m, n)), -r_mat, np.zeros(e_u_p_shape),
                      np.zeros(e_u_p_shape), -np.ones(e_u_p_shape)]])

    b_ub = np.zeros((2*(l+n+m),))

    A_eq = np.concatenate([a_mat, b_mat, np.zeros((n, 3))], axis=1)
    b_eq = curr_state

    c = np.concatenate([np.zeros((n,)), np.zeros((m,)), np.ones((3,))])

    bounds = np.array([[None, None] for _ in range(n+m)] + [[0, None], [0, None], [0, None]])

    result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    state, control, slack = result.x[:n], result.x[n:n+m], result.x[n+m:]
    return state, control, slack

def calc_p_from_opt(state, control, slack):
    return np.zeros((state.shape[0], state.shape[0]))

def example_9_1():
    Q = np.eye(2)
    R = 20*np.ones((1,))
    A = np.array([[1, 1], [0, 1]])
    B = np.array([0, 1]).reshape(-1, 1)
    x0 = np.zeros((2,))

    error = 1
    curr_p = Q
    curr_x = x0

    ps, xs, us = [curr_p], [curr_x], []

    while error > 1e-6:
        prev_p, prev_x, prev_u = calc_prev_p_matrix(curr_x, curr_p, Q, R, A, B)
        diff = curr_p - prev_p
        error = np.linalg.norm(diff)

        ps.append(prev_p)
        xs.append(prev_x)
        us.append(prev_u)

        curr_p = prev_p
        curr_x = prev_x

    return ps.reverse(), xs.reverse(), us.reverse()

if __name__ == "__main__":
    example_9_1()
