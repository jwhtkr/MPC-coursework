

import numpy as np
import numpy.linalg as la
import time


def calc_alphastar(G, w, z, delta_star, active_ind):
    alpha_star = 1
    alpha_star_ind = np.inf
    for ind, is_active in enumerate(active_ind):
        if not is_active:
            Gi, wi = G[ind,:], w[ind]
            if Gi.T @ delta_star > 0:
                alpha = (wi - Gi.T @ z)/(Gi.T @ delta_star)
            else:
                alpha = np.inf
            if alpha < alpha_star:
                alpha_star = alpha
                alpha_star_ind = ind
    return (alpha_star, alpha_star_ind)

def calc_deltastar(Ga, H, q, z):
    ua = np.asarray(calc_ua(Ga, H, q, z))
    try:
        tmp = H @ z + Ga.T @ ua + q
    except ValueError:
        tmp = H @ z + Ga.T * ua + q
    return -la.inv(H) @ tmp, ua

# def calc_ua(Ga, H, z):
#     return -la.inv(Ga @ Ga.T) @ Ga @ H @ z

def calc_ua(Ga, H, q, z):
    M = Ga @ la.inv(H) @ Ga.T
    rhs = -Ga @ la.inv(H) @ (q + H @ z)

    return la.inv(M) @ rhs if len(M.shape) > 1 else 1/M * rhs

def step(H, q, G, W, z, active_ind):
    new_active_ind = [item for item in active_ind]
    Ga = G[active_ind, :]
    delta_star, ua = calc_deltastar(Ga, H, q, z)
    if np.all(delta_star == 0):
        if np.all(ua >= 0):
            return True, z, new_active_ind
        else:
            inds = [i for i, ind in enumerate(active_ind) if ind]
            ind_to_remove = inds[np.argmin(ua)]
            new_active_ind[ind_to_remove] = False
            return False, z, new_active_ind
    else:
        alpha_star, blocking_ind = calc_alphastar(G, w, z, delta_star, active_ind)
        try:
            new_active_ind[blocking_ind] = True
        except IndexError:
            pass
        finally:
            return False, z + alpha_star * delta_star, new_active_ind

def solve(H, q, G, w, z0=None, max_itr=1000):
    if z0 is None:
        raise ValueError("Only warm-started solving is currently supported.")

    active_ind_0 = G @ z0 == w
    active_ind_list, z_list = [active_ind_0], [z0]

    itr = 0
    stop = False
    while not stop and itr < max_itr:
        stop, z, active_ind = step(H, q, G, w, z_list[itr], active_ind_list[itr])
        active_ind_list.append(active_ind)
        z_list.append(z)
        itr += 1

    return z_list, active_ind_list, itr

if __name__ == "__main__":
    G = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    w = np.array([-1, 1, 1, 3])
    H = np.eye(2)
    q = np.zeros((2,))
    z0 = np.array([1, 2])

    start_time = time.time()
    zs, active_inds, itr = solve(H, q, G, w, z0)

    print("Optimal z: {}\tfound in {} iterations ({} sec)".format(zs[-1], itr, time.time()-start_time))
