import numpy as np
import pandas as pd
import utils
import scipy.linalg as slin
import igraph
from skfda.representation.basis import Fourier
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from train import Adam


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


def multiDAG_functional(X, lambda1, rho, P_id, P_all, max_iter=200, alpha_max=1e+16, h_tol=1e-6, G_true=None):
    def _bounds():
        bounds = []
        for _ in range(2):
            for l in range(L):
                for i in range(P[l] * K):
                    for j in range(P[l] * K):
                        ni, nj = int(i / K), int(j / K)
                        if ni == nj:
                            bounds.append((0, 0))
                        else:
                            bounds.append((0, None))
        return bounds

    def _adj(gt):
        G = {}
        iter = 0
        for l in range(L):
            G[l] = np.zeros((P[l] * K, P[l] * K))
            for i in range(P[l] * K):
                for j in range(P[l] * K):
                    G[l][i, j] = gt[iter]
                    iter += 1
        return G

    def _vec(G1):
        g = []
        for l in range(L):
            for i in range(P[l] * K):
                for j in range(P[l] * K):
                    g.append(G1[l][i, j])
        return np.array(g)

    def _loss(G):
        G_Gloss = {}
        loss = 0
        for l in range(L):
            G_Gloss[l] = np.zeros((P[l] * K, P[l] * K))
            M = X[l] @ G[l]
            R = X[l] - M
            loss += 0.5 / X[l].shape[0] * (R ** 2).sum()
            G_Gloss[l] = - 1.0 / X[l].shape[0] * X[l].T @ R
        return loss, G_Gloss

    def _f(G):
        fG = {}
        G_fG = {}
        for l in range(L):
            fG[l] = np.zeros((P[l], P[l]))
            G_fG[l] = np.zeros((P[l], P[l], P[l] * K, P[l] * K))
            for i in range(P[l]):
                for j in range(P[l]):
                    fG[l][i, j] = np.sum((G[l][i * K:(i + 1) * K, j * K:(j + 1) * K]) ** 2) * 0.5
                    G_fG[l][i, j, i * K:(i + 1) * K, j * K:(j + 1) * K] = G[l][i * K:(i + 1) * K, j * K:(j + 1) * K]
        return fG, G_fG

    def _h(G):
        fG, G_fG = _f(G)
        h = 0
        p = 0
        E = {}
        G_hG = {}
        G_pG = {}
        for l in range(L):
            G_hG[l] = np.zeros((P[l] * K, P[l] * K))
            G_pG[l] = np.zeros((P[l] * K, P[l] * K))
            E[l] = slin.expm(fG[l])
            h += np.trace(E[l]) - P[l]
            G_hG[l] = (E[l].T.reshape(P[l] * P[l]) @ G_fG[l].reshape(P[l] * P[l], P[l] * K * P[l] * K)).reshape(P[l] * K, P[l] * K)

        re_id = {}
        for l in range(L):
            re_id[l] = {}
            for i in range(P[l]):
                re_id[l][P_id[l][i]] = i

        def link(A):
            lA = {}
            dA = {}
            for l in range(L):
                P = A[l].shape[0]
                lA[l] = np.zeros((P, P))
                dA[l] = np.zeros((P, P, P, P))
                pA = {}
                pA[0] = np.identity(P)
                for i in range(1, P):
                    pA[i] = pA[i - 1] @ A[l]
                for i in range(P):
                    lA[l] = lA[l] + pA[i]
                for i in range(1, P):
                    for r in range(i):
                        dA[l] += np.einsum('ik,lj->ijkl', pA[r], pA[i - r - 1])
            return lA, dA

        def sigmoid(X):
            if X > 0:
                return 1 / (1 + np.exp(-X))
            else:
                return np.exp(X) / (np.exp(X) + 1)

        def A(X, Y): # eX * (eX - eY) / (eX + 1) ** 3 / (eY + 1)
            eX, eY = np.exp(X), np.exp(Y)
            _eX, _eY = np.exp(-X), np.exp(-Y)
            return 1 / (eX + 1) / (eY + 1) / (1 + _eX) ** 2 - 1 / (eX + 1) ** 2 / (_eY + 1) / (_eX + 1)

        if rho > 0:
            lA, dA = link(fG)
            dB = {}
            for l in range(L):
                dB[l] = np.einsum('xyij,ijkl->xykl', dA[l], G_fG[l])
            for l in range(L):
                for _l in range(L):
                    if l == _l:
                        continue
                    for ni in range(P_all):
                        for nj in range(P_all):
                            if ni == nj:
                                continue
                            if (ni in re_id[l]) and (nj in re_id[l]) and (ni in re_id[_l]) and (nj in re_id[_l]):
                                li, lj = re_id[l][ni], re_id[l][nj]
                                _li, _lj = re_id[_l][ni], re_id[_l][nj]
                                X = 10 * (lA[l][li, lj] - lA[l][lj, li])
                                Y = 10 * (lA[_l][_li, _lj] - lA[_l][_lj, _li])
                                p += (sigmoid(X) - sigmoid(Y)) ** 2 * rho
                                G_pG[l] += 40 * rho * A(X, Y) * (dB[l][li, lj] - dB[l][lj, li])
        return h, p, G_hG, G_pG

    def _func(gt):
        G = _adj(gt)
        loss, G_Gloss = _loss(G)
        h, p, G_hG, G_pG = _h(G)
        g_loss = loss + h * (beta + 0.5 * alpha * h) + p
        G_Gsmooth = {}
        g_loss += lambda1 * np.sum(np.abs(gt))
        for l in range(L):
            G_Gsmooth[l] = G_Gloss[l] + (alpha * h + beta) * G_hG[l] + lambda1 + G_pG[l]
        g_grad = _vec(G_Gsmooth)
        return g_loss, g_grad

    # def _grad(gt):
    #     G = _adj(gt)
    #     loss, G_Gloss = _loss(G)
    #     h, p, G_hG, G_pG = _h(G)
    #     g_loss = loss + h * (beta + 0.5 * alpha * h) + p
    #     G_Gsmooth = {}
    #
    #     g_loss += lambda1 * np.sum(np.abs(gt))
    #     for l in range(L):
    #         G_Gsmooth[l] = G_Gloss[l] + (alpha * h + beta) * G_hG[l] * np.sign(G) + lambda1 + G_pG[l]
    #     g_grad = _vec(G_Gsmooth)
    #     return g_grad

    def _debug(gt):
        G = _adj(gt)
        loss, G_Gloss = _loss(G)
        h, p, G_hG, G_pG = _h(G)
        print("loss:%f, p:%f, h:%f, alpha:%f" % (loss, p, h, alpha))

    def _grad(gt):
        G = _adj(gt)
        loss, G_Gloss = _loss(G)
        h, p, G_hG, G_pG = _h(G)
        G_Gsmooth = {}
        for l in range(L):
            G_Gsmooth[l] = G_Gloss[l] + (alpha * h + beta) * G_hG[l] + lambda1 * np.sign(G[l]) + G_pG[l]
        g_obj = _vec(G_Gsmooth)
        return g_obj

    def _train(optimizer):
        for _ in range(1000):
            optimizer.update(_grad(optimizer.params))
        return optimizer.params

    L = len(X)
    N, P = [], []
    Empty = {}
    K = 0
    G = {}
    for l in range(L):
        [Nl, Pl, K] = np.shape(X[l])
        N.append(Nl)
        P.append(Pl)
        X[l] = X[l].reshape((Nl, Pl * K))
        Empty[l] = np.zeros((Pl * K, Pl * K))
        G[l] = np.random.rand(Pl * K, Pl * K)
    bounds = _bounds()
    gt_est = _vec(Empty)
    alpha, beta, h = 1, 1, np.inf
    if G_true is not None:
        gt_true = _vec(G_true)
        _debug(gt_true)
    for _ in range(max_iter):
        while alpha < alpha_max:
            # sol = sopt.minimize(_func, gt_est, method='l-bfgs-b', jac=True, bounds=bounds)
            # gt_new = sol.x
            optimizer = Adam(gt_est=gt_est)
            gt_new = _train(optimizer)
            G = _adj(gt_new)
            h_new, _, _, _ = _h(G)
            # _func(gt_new)
            # _debug(gt_new)
            if h_new > 0.25 * h:
                alpha *= 10
            else:
                break
        gt_est, h = gt_new, h_new
        beta += alpha * h
        if alpha >= alpha_max or h <= h_tol:
            break
    G_est = _adj(gt_est)
    E_est = {}
    for l in range(L):
        E_est[l] = np.zeros((P[l], P[l]))
        G_est[l] = G_est[l]
        for i in range(P[l] * K):
            for j in range(P[l] * K):
                E_est[l][int(i / K), int(j / K)] += G_est[l][i, j] ** 2
        for i in range(P[l]):
            for j in range(P[l]):
                E_est[l][i, j] = np.sqrt(E_est[l][i, j])
    return E_est, G_est


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
        E_est, G_est = multiDAG_functional(a.copy(), lambda1=np.log(P) / N / 4, rho=0.1, P_id=npid, P_all=P)
        for l in range(n_task):
            np.savetxt('./graph/E_est_multi0_%i,task_%i.csv' % (n_task, l + n_task * rep_id), E_est[l])
            np.savetxt('./graph/W_true_multi0_%i,task_%i.csv' % (n_task, l + n_task * rep_id), W_true[l + rep_id * n_task])
            E_est[l][E_est[l] < 0.5] = 0
            E_est[l][E_est[l] > 0.1] = 1
            acc = utils.count_accuracy(B_true=E_true[l + rep_id * n_task], B_est=E_est[l])
            result_pd.loc[niter] = [N, K, P, l, acc['fdr'], acc['tpr'], 'MULTITASK-0', rep_id]
            niter += 1
    result_pd.to_csv('./result/N=%i,K=%i,P=%i,L=%i.csv' % (N, K, P, n_task))


if __name__ == '__main__':
    single_test(10, 2, 10, 4, 1)