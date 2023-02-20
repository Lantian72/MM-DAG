import numpy as np
import pandas as pd

import utils
import scipy.linalg as slin
import igraph
from skfda.representation.basis import Fourier
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from train import Adam
from multiprocessing import freeze_support
from multiprocessing import Pool


def generate_multiDAG(N=1000, P=10, K=2, s=20, L=5, T=100, P_task=None, P_id=None, N_task=None):
    import random
    utils.set_random_seed(233)
    if P_task is None:
        P_task = []
        P_id = {}
        for l in range(L):
            P_task.append(random.randint(int(P / 2), P))
            P_id[l] = random.sample(range(P), P_task[l])
    graph_type = 'ER'
    T_true = utils.simulate_dag(P, s, graph_type)
    igraph.Graph.Adjacency(T_true)
    T_true = slin.expm(T_true) > 0
    for p in range(P):
        T_true[p, p] = 0

    TW_true = utils.simulate_parameter(T_true)
    E_true = {}
    W_true = {}
    G_true = {}
    a_true = {}
    g = {}
    h = {}
    fourier_basis = Fourier((0, 1), n_basis=K, period=1)
    basis = fourier_basis(np.arange(0, 1, 1 / T))
    for l in range(L):
        # S_true = utils.simulate_dag(P, int(P * (P - 1) / 2), graph_type) * T_true
        # T_true = utils.simulate_dag(P, s, graph_type)
        E_true[l] = np.zeros((P_task[l], P_task[l]))
        W_true[l] = np.zeros((P_task[l], P_task[l]))
        G_true[l] = np.zeros((P_task[l] * K, P_task[l] * K))
        for p in range(P_task[l]):
            for _p in range(P_task[l]):
                E_true[l][p, _p] = T_true[P_id[l][p], P_id[l][_p]] * (random.uniform(0, 1) < 1)
                # W_true[l][p, _p] = TW_true[P_id[l][p], P_id[l][_p]]
                # G_true[l][p * K:(p + 1) * K, _p * K:(_p + 1) * K] = np.identity(K) * TW_true[P_id[l][p], P_id[l][_p]]
        W_true[l] = utils.simulate_parameter(E_true[l])

    if N_task is None:
        N_task = []
        for l in range(L):
            N_task.append(N)
    for l in range(L):
        G = igraph.Graph.Adjacency(E_true[l])
        ordered_vertices = G.topological_sorting()
        g[l] = np.zeros((N_task[l], P_task[l], T))
        h[l] = np.zeros((N_task[l], P_task[l], T))
        a_true[l] = np.zeros((N, P_task[l], K))
        for i in range(N_task[l]):
            delta_i = np.zeros((P_task[l], K))
            for j in ordered_vertices:
                parents = G.neighbors(j, mode=igraph.IN)
                mean = np.zeros(K)
                for k in parents:
                    mean += (delta_i[k, :] * W_true[l][k, j]).reshape(K)
                delta_i[j, :] = np.random.multivariate_normal(mean=mean, cov=np.identity(K))
                if P_id[l][j] <= P / 2:
                    delta_i[j, 1] = 0
            for j in range(P_task[l]):
                for k in range(K):
                    if P_id[l][j] <= P / 2:
                        g[l][i, j, :] += np.full((T, ), delta_i[j, k])
                    else:
                        g[l][i, j, :] += (delta_i[j, k] * basis[k, :]).reshape((T,))
                for t in range(T):
                    h[l][i, j, t] = np.random.normal(loc=g[l][i, j, t], scale=0.01)
            a_true[l][i, :, :] = delta_i
    return g, h, E_true, W_true, T_true, P_id, G_true, a_true


def net_fpca(h, K):
    L = len(h)
    a = {}
    v = {}
    for l in range(L):
        [N, P, T] = np.shape(h[l])
        a[l] = np.zeros((N, P, K))
        v[l] = np.zeros((P, K, T))
        for j in range(P):
            fdata_ij = FDataGrid(h[l][:, j, :], np.arange(0, 1, 1 / T))
            fpca_grid = FPCA(K)
            v[l][j, :, :] = fpca_grid.fit(fdata_ij).components_.data_matrix.reshape(K, T)
            fpca_grid = FPCA(K)
            a[l][:, j, :] = fpca_grid.fit_transform(fdata_ij)
    return a, v


def multiDAG_functional(X, lambda1, rho, P_id, P_all, max_iter=200, alpha_max=1e+16):
    def _bounds():
        bounds = {}
        for l in range(L):
            bounds[l] = np.zeros((P[l] * K, P[l] * K))
            for i in range(P[l] * K):
                for j in range(P[l] * K):
                    ni, nj = int(i / K), int(j / K)
                    #if (P_id[l][ni] <= 4 and 5 <= P_id[l][nj] <= 13) or (5 < P_id[l][ni] <= 13 and P_id[l][ni] != P_id[l][nj] and 5 < P_id[l][nj]):
                    if ni != nj:
                        bounds[l][i, j] = 0
                    else:
                        bounds[l][i, j] = 1
        return bounds
    def _adj(gt):
        G = {}
        T = {}
        iter = 0
        for l in range(L):
            G[l] = np.zeros((P[l] * K, P[l] * K))
            for i in range(P[l] * K):
                for j in range(P[l] * K):
                    G[l][i, j] = gt[iter]
                    iter += 1
        for l in range(L):
            T[l] = np.zeros((P[l] * K, P[l] * K))
            for i in range(P[l] * K):
                for j in range(P[l] * K):
                    T[l][i, j] = gt[iter]
                    iter += 1
        return G, T

    def _vec(G, T):
        gt = []
        for l in range(L):
            for i in range(P[l] * K):
                for j in range(P[l] * K):
                    gt.append(G[l][i, j])
        for l in range(L):
            for i in range(P[l] * K):
                for j in range(P[l] * K):
                    gt.append(T[l][i, j])
        return np.array(gt)

    def _loss(G, T):
        G_Gloss = {}
        G_Tloss = {}
        loss = 0
        for l in range(L):
            G_Gloss[l] = np.zeros((P[l] * K, P[l] * K))
            G_Tloss[l] = np.zeros((P[l] * K, P[l] * K))
            Gbar = G[l] * T[l]
            M = X[l] @ Gbar
            R = X[l] - M
            loss += 0.5 / X[l].shape[0] * (R ** 2).sum()
            G_Gloss[l] = - 1.0 / X[l].shape[0] * X[l].T @ R * T[l]
            G_Tloss[l] = - 1.0 / X[l].shape[0] * X[l].T @ R * G[l]
        return loss, G_Gloss, G_Tloss

    def _f(G, T):
        fT = {}
        G_fT = {}
        G_fG = {}
        for l in range(L):
            fT[l] = np.zeros((P[l], P[l]))
            G_fG[l] = np.zeros((P[l], P[l], P[l] * K, P[l] * K))
            G_fT[l] = np.zeros((P[l], P[l], P[l] * K, P[l] * K))
            for i in range(P[l]):
                for j in range(P[l]):
                    fT[l][i, j] = np.sum((T[l][i * K:(i + 1) * K, j * K:(j + 1) * K]
                                         * G[l][i * K:(i + 1) * K, j * K:(j + 1) * K]) ** 2) * 0.5
                    for _i in range(i * K, (i + 1) * K):
                        for _j in range(j * K, (j + 1) * K):
                            G_fT[l][i, j, _i, _j] = T[l][_i, _j] * G[l][_i, _j] ** 2
                            G_fG[l][i, j, _i, _j] = G[l][_i, _j] * T[l][_i, _j] ** 2
        return fT, G_fT, G_fG

    def _h(G, T):
        fT, G_fT, G_fG = _f(G, T)
        h = 0
        G_hT = {}
        G_hG = {}
        for l in range(L):
            G_hT[l] = np.zeros((P[l] * K, P[l] * K))
            G_hG[l] = np.zeros((P[l] * K, P[l] * K))
            E = slin.expm(fT[l])
            h += np.trace(E) - P[l]
            G_hT[l] = (E.T.reshape(P[l] * P[l]) @ G_fT[l].reshape(P[l] * P[l], P[l] * K * P[l] * K)).reshape(P[l] * K, P[l] * K)
            G_hG[l] = (E.T.reshape(P[l] * P[l]) @ G_fG[l].reshape(P[l] * P[l], P[l] * K * P[l] * K)).reshape(P[l] * K, P[l] * K)
        return h, G_hG, G_hT

    def _func(gt):
        G, T = _adj(gt)
        loss, G_Gloss, G_Tloss = _loss(G, T)
        h, G_hG, G_hT = _h(G, T)
        print("loss:%f, h:%f" % (loss, h))

    def _grad(gt):
        G, T = _adj(gt)
        loss, G_Gloss, G_Tloss = _loss(G, T)
        h, G_hG, G_hT = _h(G, T)
        G_Gsmooth = {}
        G_Tsmooth = {}
        for l in range(L):
            G_Gsmooth[l] = G_Gloss[l] + (alpha * h + beta) * G_hG[l] + lambda1 * np.sign(G[l])
            G_Tsmooth[l] = G_Tloss[l] + (alpha * h + beta) * G_hT[l] - 2 * rho * (np.ones((P[l] * K, P[l] * K)) - T[l]) # + lambda1 * np.sign(T[l])

        for l in range(L):
            for i in range(P[l] * K):
                for j in range(P[l] * K):
                    ni, nj = int(i / K), int(j / K)
                    if bounds[l][i, j] == 1:
                        G_Gsmooth[l][i, j] += 2000 * G[l][i, j]

        for l in range(L):
            for i in range(P[l] * K):
                for j in range(P[l] * K):
                    ni, nj = int(i / K), int(j / K)
                    if bounds[l][i, j] == 1:
                        G_Tsmooth[l][i, j] += 2000 * T[l][i, j]
        re_id = {}
        for l in range(L):
            re_id[l] = {}
            for i in range(P[l]):
                re_id[l][P_id[l][i]] = i
        for l in range(L):
            for _l in range(L):
                for ni in range(P_all):
                    for nj in range(P_all):
                        if (ni in re_id[l]) and (nj in re_id[l]) and (ni in re_id[_l]) and (nj in re_id[_l]):
                            li, lj = re_id[l][ni], re_id[l][nj]
                            _li, _lj = re_id[_l][ni], re_id[_l][nj]
                            G_Tsmooth[l][li * K:(li + 1) * K, lj * K:(lj + 1) * K] += \
                                20 * (T[l][li * K:(li + 1) * K, lj * K:(lj + 1) * K]
                                       - T[_l][_li * K:(_li + 1) * K, _lj * K:(_lj + 1) * K])
        g_obj = _vec(G_Gsmooth, G_Tsmooth)
        return g_obj

    def _train(optimizer):
        for _ in range(1000):
            optimizer.update(_grad(optimizer.params))
        return optimizer.params

    L = len(X)
    N, P = [], []
    K = 0
    G = {}
    T = {}
    for l in range(L):
        [Nl, Pl, K] = np.shape(X[l])
        N.append(Nl)
        P.append(Pl)
        X[l] = X[l].reshape((Nl, Pl * K))
        G[l] = np.zeros((Pl * K, Pl * K))
        T[l] = np.zeros((Pl * K, Pl * K))
    bounds = _bounds()
    gt_est = _vec(G, T)
    alpha, beta, h = 1, 1, np.inf
    for _ in range(max_iter):
        while alpha < alpha_max:
            optimizer = Adam(gt_est=_vec(G, T))
            gt_new = _train(optimizer)
            G, T = _adj(gt_new)
            h_new, _, _ = _h(G, T)
            _func(gt_new)
            if h_new > 0.25 * h:
                alpha *= 10
                # rho = rho * 1.3
            else:
                break
        gt_est, h = gt_new, h_new
        beta += alpha * h
        if alpha >= alpha_max:
            break
    G_est, T_est = _adj(gt_est)
    E_est = {}
    optimizer = Adam(gt_est=_vec(G_est, T_est))
    gt_new = _train(optimizer)
    G_est, T_est = _adj(gt_new)
    for l in range(L):
        E_est[l] = np.zeros((P[l], P[l]))
        G_est[l] = G_est[l] * T_est[l]
        for i in range(P[l] * K):
            for j in range(P[l] * K):
                E_est[l][int(i / K), int(j / K)] += G_est[l][i, j] ** 2
        for i in range(P[l]):
            for j in range(P[l]):
                E_est[l][i, j] = np.sqrt(E_est[l][i, j] / K)
    # E_est[abs(E_est) < 0.01] = 0
    return E_est, G_est, T_est


def single_test(N=None, K=None, P=None, n_task=None, rep=None):
    if rep is None:
        rep = 1
    g, h, E_true, W_true, T_true, P_id, G_true, a_true = generate_multiDAG(N=N, L=n_task * rep, P=P, K=K, s=14)
    acc = {}
    niter = 0
    result_pd = pd.DataFrame(columns=['N', 'K', 'P', 'L', 'fdr', 'tpr', 'method', 'rep_id'])
    for rep_id in range(rep):
        nh = {}
        npid = {}
        for l in range(rep_id * n_task, (rep_id + 1) * n_task):
            nh[l - rep_id * n_task] = h[l].copy()
            npid[l - rep_id * n_task] = P_id[l].copy()
        a, v = net_fpca(nh, K)
        E_est, _, _ = multiDAG_functional(a.copy(), lambda1=np.log(P) / (N * n_task), rho=0.1, P_id=npid, P_all=P)
        for l in range(n_task):
            np.savetxt('./result/graph/uncertain/E_est_%i,task_%i.csv' % (n_task, l + n_task * rep_id), E_est[l])
            np.savetxt('./result/graph/uncertain/W_true_%i,task_%i.csv' % (n_task, l + n_task * rep_id), W_true[l + rep_id * n_task])
            E_est[l][E_est[l] < 0.5] = 0
            E_est[l][E_est[l] > 0.1] = 1
            acc = utils.count_accuracy(B_true=E_true[l + rep_id * n_task], B_est=E_est[l])
            result_pd.loc[niter] = [N, K, P, l, acc['fdr'], acc['tpr'], 'MULTITASK-UN', rep_id]
            niter += 1
    result_pd.to_csv('./result/uncertain/8N=%i,K=%i,P=%i,L=%i.csv' % (N, K, P, n_task))


if __name__ == '__main__':
    freeze_support()
    ps = Pool(13)
    N_level = [15, 30, 50, 100, 200, 400, 800]
    N_task_level = [1, 2, 4, 8, 16, 32]
    # single_test(800, 2, 10, 2, 1)
    # for N in N_level:
    #     ps.apply_async(single_test, args=(N, 2, 10, 4, 8))
    for tasks in N_task_level:
        ps.apply_async(single_test, args=(10, 2, 10, tasks, int(32 / tasks)))
    ps.close()
    ps.join()