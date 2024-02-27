

import numpy as np
import numpy.linalg as la
import time
import math
import scipy.sparse as sparse


def calc_initial_dual(obj, eq_mat, eq_vec, ineq, tau_initial, initial_primal):
    z0 = initial_primal
    gradient_f = obj.gradient(z0).flatten()
    s_vec = -ineq(z0).flatten()
    if np.any(s_vec <= 0):
        raise RuntimeError("Initial solution is not in the interior of the "
                           "feasible set.")
    s_mat = np.diag(s_vec)

    g_mat = ineq.jacobian(z0)
    tau_vec = np.repeat(tau_initial, len(s_vec))

    A = sparse.bmat([[eq_mat.T, g_mat.T], [None, s_mat]])
    A = A.toarray()
    b = np.concatenate([-gradient_f, tau_vec])

    sol = la.lstsq(A, b, rcond=None)
    v_u_vec = sol[0]
    # v_u_vec = la.solve(A, b)

    v_vec, u_vec = v_u_vec[:eq_mat.shape[0]], v_u_vec[eq_mat.shape[0]:]
    if np.any(u_vec <= 0):
        raise RuntimeError("Dual u variables were not all positive.")

    return v_vec, u_vec, s_vec

def calc_duality_measure(s_vec, u_vec):
    if len(s_vec) != len(u_vec):
        raise ValueError("s and u were not the same length")

    return np.dot(s_vec, u_vec)/len(s_vec)

def calc_max_step(s_vec, u_vec, s_step, u_step, tol=1e-6):
    niter = math.ceil(abs(math.log2(tol)))
    c_vec, c_step = np.concatenate([s_vec, u_vec]), np.concatenate([s_step, u_step])

    if np.all(c_vec + c_step > 0):
        return 1.

    feasible_step = 0.
    for i in range(1, niter+1):
        curr_step = feasible_step + 2**(-i)
        if np.all(c_vec + curr_step * c_step > 0):
            feasible_step = curr_step

    return feasible_step

def calc_max_step_backtracking(s_vec, u_vec, s_step, u_step, tol=1e-6, alpha=0.2, beta=0.6):
    pass

def calc_step_direction(obj, eq_mat, eq_vec, ineq, z_vec, v_vec, u_vec, s_vec, centering):
    h_mat = obj.hessian(z_vec) + u_vec @ ineq.hessian(z_vec)
    g_mat = ineq.jacobian(z_vec)
    f_grad = obj.gradient(z_vec)
    s_mat = np.diag(s_vec)
    u_mat = np.diag(u_vec)

    a_blocks = [[h_mat, eq_mat.T, g_mat.T, None], [eq_mat, None, None, None],
                [g_mat, None, None, np.eye(len(s_vec))], [None, None, s_mat, u_mat]]
    b_blocks = [-(f_grad + eq_mat.T @ v_vec + g_mat.T @ u_vec),
                -(eq_mat @ z_vec - eq_vec),
                -(ineq(z_vec) + s_vec),
                -(s_mat @ u_vec - centering)]

    A = sparse.bmat(a_blocks)
    A = A.toarray()
    b = np.concatenate(b_blocks)

    direction = la.solve(A, b)
    z, v, u, s = (direction[:len(z_vec)],
                  direction[len(z_vec):len(z_vec)+len(v_vec)],
                  direction[len(z_vec)+len(v_vec):len(z_vec)+len(v_vec)+len(u_vec)],
                  direction[len(z_vec)+len(v_vec)+len(u_vec):])

    return z, v, u, s


def step(objective, eq_mat, eq_vec, ineq_constrs, z_vec, v_vec, u_vec, s_vec, sigma=0.5, tol=1e-6):
    mu = calc_duality_measure(s_vec, u_vec)
    del_z, del_v, del_u, del_s = calc_step_direction(objective, eq_mat, eq_vec, ineq_constrs,
                                                     z_vec, v_vec, u_vec, s_vec, sigma*mu)
    h = calc_max_step(s_vec, u_vec, del_s, del_u, tol)

    z, v, u, s = z_vec+h*del_z, v_vec+h*del_v, u_vec+h*del_u, s_vec+h*del_s

    if la.norm(h*del_z) <= tol:
        should_stop = True
    else:
        should_stop = False

    return should_stop, z, v, u, s


if __name__ == "__main__":
    G = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    w = np.array([-1, 1, 1, 3])
    H = np.eye(2)
    q = np.zeros((2,))
    z0 = np.array([1, 1])

    def objective(z):
        return 0.5 * z @ H @ z + q @ z

    def objective_gradient(z):
        return z @ H + q

    def objective_hessian(z):
        return H

    def ineq_constrs(z):
        return G @ z - w

    def ineq_constrs_jacobian(z):
        return G

    def ineq_constrs_hessian(z):
        return np.zeros_like(G)

    objective.gradient = objective_gradient
    objective.hessian = objective_hessian

    ineq_constrs.jacobian = ineq_constrs_jacobian
    ineq_constrs.hessian = ineq_constrs_hessian

    A = np.empty((0,2))
    b = np.empty((0,))

    v0, u0, s0 = calc_initial_dual(objective, A, b, ineq_constrs, 1, z0)

    zs, vs, us, ss = [z0], [v0], [u0], [s0]

    start_time = time.time()
    stop = False
    itr = 0
    while not stop and itr < 1000:
        stop, z, v, u, s = step(objective, A, b, ineq_constrs, zs[itr], vs[itr],
                                us[itr], ss[itr])
        zs.append(z); vs.append(v); us.append(u); ss.append(s)
        itr += 1

    print("Optimal z: {}\tfound in {} iterations ({} sec)".format(zs[-1], itr, time.time()-start_time))
