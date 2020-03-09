import numpy as np
from scipy import integrate
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.integrate import quad

iterations = 1000

n = 1000000
pi0 = 0.5
pi1 = 0.5
mu0 = np.array([0, 0, 0])
mu1 = np.array([1, 1, 1])
cov0 = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1]))


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def gen_3x3_cov():
    # If the covariance matrix we produce isn't valid, we keep trying until we get one that works.
    valid = False
    while valid == False:
        sd = np.sqrt(np.array(np.random.random(3)))
        rho = np.array(np.random.random(3)) * np.random.choice(
            a=[-1, 1], size=3, replace=True
        )

        sigma = np.diag(sd ** 2)

        ut = np.triu_indices(n=3, k=1)
        sigma[ut] = np.array(
            (rho[0] * sd[0] * sd[1], rho[1] * sd[0] * sd[2], rho[2] * sd[1] * sd[2])
        )
        lt = np.tril_indices(n=3, k=-1)
        sigma[lt] = sigma.T[lt]
        # print(is_pos_def(sigma))
        valid = is_pos_def(sigma)
    return sigma


# Single-Feature Loss


def single_feature_loss(mu0, mu1, cov0, cov1, pi0, pi1, n, dim):
    class0 = np.random.normal(mu0[dim], cov0[dim, dim], n)
    class1 = np.random.normal(mu1[dim], cov1[dim, dim], n)
    loss = pi0 * (
        np.mean(
            norm.pdf(class0, mu1[dim], np.sqrt(cov1[dim, dim]))
            > norm.pdf(class0, mu0[dim], np.sqrt(cov0[dim, dim]))
        )
    ) + pi1 * (
        np.mean(
            norm.pdf(class1, mu0[dim], np.sqrt(cov0[dim, dim]))
            > norm.pdf(class1, mu1[dim], np.sqrt(cov1[dim, dim]))
        )
    )
    return loss


# Two-Feature Loss


def two_feature_loss(mu0, mu1, cov0, cov1, pi0, pi1, n, dim1, dim2):
    class0 = np.random.multivariate_normal(
        mu0[[dim1, dim2]], cov0[np.ix_([dim1, dim2], [dim1, dim2])], n
    )
    class1 = np.random.multivariate_normal(
        mu1[[dim1, dim2]], cov1[np.ix_([dim1, dim2], [dim1, dim2])], n
    )
    loss = pi0 * (
        np.mean(
            multivariate_normal.pdf(
                class0,
                mean=mu1[[dim1, dim2]],
                cov=cov1[np.ix_([dim1, dim2], [dim1, dim2])],
            )
            > multivariate_normal.pdf(
                class0,
                mean=mu0[[dim1, dim2]],
                cov=cov0[np.ix_([dim1, dim2], [dim1, dim2])],
            )
        )
    ) + pi1 * (
        np.mean(
            multivariate_normal.pdf(
                class1,
                mean=mu0[[dim1, dim2]],
                cov=cov0[np.ix_([dim1, dim2], [dim1, dim2])],
            )
            > multivariate_normal.pdf(
                class1,
                mean=mu1[[dim1, dim2]],
                cov=cov1[np.ix_([dim1, dim2], [dim1, dim2])],
            )
        )
    )
    return loss


# Compute true conditional entropy for multivariate normal class conditional distribution.


def calc_H_X(mean0, mean1, cov0, cov1, pi0, pi1):

    # H_X = multidimensional integral on the sum of pi_0 * pdf0(x) + pi_1 * pdf1(x)
    def integrand(*args):
        rv0 = multivariate_normal(mean=args[-6], cov=args[-4])
        rv1 = multivariate_normal(mean=args[-5], cov=args[-3])
        f_x = args[-2] * rv0.pdf(args[:-6]) + args[-1] * rv1.pdf(args[:-6])
        return -f_x * np.log(f_x) / np.log(2.0)

    sd0 = np.sqrt(np.diag(cov0))
    sd1 = np.sqrt(np.diag(cov1))
    mins = np.min([mean0 - 4 * sd0, mean1 - 4 * sd1], axis=0)
    maxes = np.max([mean0 + 4 * sd0, mean1 + 4 * sd1], axis=0)
    limits = np.vstack((mins, maxes)).T

    res, err = integrate.nquad(
        integrand, limits, args=(mean0, mean1, cov0, cov1, pi0, pi1)
    )

    return res


def calc_H_X_Y(cov0, cov1, pi0, pi1):

    # H_XY = D / 2 * (1 + np.log(2 * np.pi)) + 1 / 2 * np.log(np.linalg.det(cov))
    # Here we use pi0 and pi1 to get a weighted average of H_X_Y_(0/1).
    D = cov0.shape[0]
    H_X_Y_0 = D / 2 * (1 + np.log(2 * np.pi)) + 1 / 2 * np.log(np.linalg.det(cov0))
    H_X_Y_1 = D / 2 * (1 + np.log(2 * np.pi)) + 1 / 2 * np.log(np.linalg.det(cov1))

    return (pi0 * H_X_Y_0 + pi1 * H_X_Y_1) / np.log(2.0)


def two_feature_H_Y_X(mu0, mu1, cov0, cov1, pi0, pi1, dim1, dim2):
    mean0 = mu0[[dim1, dim2]]
    mean1 = mu1[[dim1, dim2]]
    reduced_cov0 = cov0[np.ix_([dim1, dim2], [dim1, dim2])]
    reduced_cov1 = cov1[np.ix_([dim1, dim2], [dim1, dim2])]
    H_Y = -(pi0 * np.log(pi0) + pi1 * np.log(pi1)) / np.log(2.0)
    H_X = calc_H_X(mean0, mean1, reduced_cov0, reduced_cov1, pi0, pi1)
    H_X_Y = calc_H_X_Y(reduced_cov0, reduced_cov1, pi0, pi1)
    H_Y_X = H_Y - H_X + H_X_Y
    return H_Y_X


def single_feature_H_Y_Xj(mean0, mean1, cov0, cov1, pi0, pi1, j):
    sd0 = np.sqrt(cov0[j, j])
    sd1 = np.sqrt(cov1[j, j])
    min_x = np.min([mean0[j] - 4 * sd0, mean1[j] - 4 * sd1])
    max_x = np.max([mean0[j] + 4 * sd0, mean1[j] + 4 * sd1])

    def func(x):
        p = pi0 * norm.pdf(x, mean0[j], sd0) + pi1 * norm.pdf(x, mean1[j], sd1)
        return -p * np.log(p) / np.log(2.0)

    H_X = quad(func, min_x, max_x)
    # H_XY = np.log(sigma*np.sqrt(2*np.pi*np.e))
    # H_XY = np.log(sigma) + .5*(1 + np.log(2*np.pi))
    # We need a weighted sum of two H_XY where weights are .5.
    # H_XY = .5*(np.log(sd0) + np.log(sd1)) + .5*(1 + np.log(2*np.pi))
    H_XY = (
        pi0 * np.log(sd0) + pi1 * np.log(sd1) + 0.5 * (1 + np.log(2 * np.pi))
    ) / np.log(2.0)
    H_Y = -(pi0 * np.log(pi0) + pi1 * np.log(pi1)) / np.log(2.0)
    # I_XY = H_X - H_XY = H_Y - H_YX
    H_Y_Xj = H_Y - H_X[0] + H_XY
    return H_Y_Xj


loss_entropy_correlation_agree_1f = np.zeros(iterations)
loss_entropy_correlation_winners_agree_1f = np.zeros(iterations)
# loss_entropy_agree_2f = np.zeros(iterations)
# loss_importance_persists = np.zeros(iterations)
entropy_importance_persists = np.zeros(iterations)
loss_values_1f = np.zeros((iterations, cov0.shape[0]))
entropy_values_1f = np.zeros((iterations, cov0.shape[0]))
correlation_values_1f = np.zeros((iterations, cov0.shape[0]))

for i in range(iterations):

    print(f"Iteration: {i}")
    np.random.seed(123 + i)

    # Construct a new cov1 matrix.
    cov1 = gen_3x3_cov()

    # Rank individual features by minimum expected loss.
    one_feature_loss_results = list()
    dim_list = [0, 1, 2]
    for dim in dim_list:
        one_feature_loss_results.append(
            single_feature_loss(mu0, mu1, cov0, cov1, pi0, pi1, n, dim)
        )
    loss_order_1f = np.array(one_feature_loss_results).argsort()
    loss_values_1f[i, :] = np.array(one_feature_loss_results)

    # Rank individual features by minimum conditional entropy.
    one_feature_entropy_results = list()
    dim_list = [0, 1, 2]
    for dim in dim_list:
        one_feature_entropy_results.append(
            single_feature_H_Y_Xj(mu0, mu1, cov0, cov1, pi0, pi1, dim)
        )
    entropy_order_1f = np.array(one_feature_entropy_results).argsort()
    entropy_values_1f[i, :] = np.array(one_feature_entropy_results)

    # Rank individual features by correlation.
    # Make sure we get the same random sample we got for loss.
    np.random.seed(123 + i)
    # For reproducibility and to reconstruct the same data samples, construct the cov1 matrix...again.
    cov1 = gen_3x3_cov()
    one_feature_correlation_results = list()
    dim_list = [0, 1, 2]
    for dim in dim_list:
        class0 = np.random.normal(mu0[dim], cov0[dim, dim], n)
        class1 = np.random.normal(mu1[dim], cov1[dim, dim], n)
        f_j = np.concatenate((class0, class1))
        c = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
        one_feature_correlation_results.append(pearsonr(f_j, c)[0])
    # Negate the array before sorting and ranking the features.
    correlation_order_1f = (-1 * np.array(one_feature_correlation_results)).argsort()
    correlation_values_1f[i, :] = np.array(one_feature_correlation_results)

    # # Rank pairs of features by minimum expected loss.
    # two_feature_loss_results = list()
    # dim_pairs_list = [(0, 1), (0, 2), (1, 2)]
    # for dim_pair in dim_pairs_list:
    #     dim1 = dim_pair[0]
    #     dim2 = dim_pair[1]
    #     two_feature_loss_results.append(
    #         two_feature_loss(mu0, mu1, cov0, cov1, pi0, pi1, n, dim1, dim2)
    #     )
    # loss_order_2f = np.array(two_feature_loss_results).argsort()

    # # Rank pairs of features by conditional entropy.
    # two_feature_entropy_results = list()
    # dim_pairs_list = [(0, 1), (0, 2), (1, 2)]
    # for dim_pair in dim_pairs_list:
    #     dim1 = dim_pair[0]
    #     dim2 = dim_pair[1]
    #     two_feature_entropy_results.append(
    #         two_feature_H_Y_X(mu0, mu1, cov0, cov1, pi0, pi1, dim1, dim2)
    #     )
    # entropy_order_2f = np.array(two_feature_entropy_results).argsort()

    # Check for outcomes of interest.
    loss_entropy_correlation_agree_1f[i] = np.all(
        loss_order_1f == entropy_order_1f
    ) & np.all(loss_order_1f == correlation_order_1f)
    loss_entropy_correlation_winners_agree_1f[i] = (
        loss_order_1f[0] == entropy_order_1f[0]
    ) & (loss_order_1f[0] == correlation_order_1f[0])
    # loss_entropy_agree_2f[i] = np.all(loss_order_2f == entropy_order_2f)
    # loss_importance_persists[i] = (
    #     (loss_order_1f[0] == 0) & (loss_order_2f[0] in [0, 1])
    #     | (loss_order_1f[0] == 1) & (loss_order_2f[0] in [0, 2])
    #     | (loss_order_1f[0] == 2) & (loss_order_2f[0] in [1, 2])
    # )
    # entropy_importance_persists[i] = (
    #     (entropy_order_1f[0] == 0) & (entropy_order_2f[0] in [0, 1])
    #     | (entropy_order_1f[0] == 1) & (entropy_order_2f[0] in [0, 2])
    #     | (entropy_order_1f[0] == 2) & (entropy_order_2f[0] in [1, 2])
    # )

np.savetxt(
    "thm_1_testing_loss_entropy_correlation_agree_1f.csv",
    loss_entropy_correlation_agree_1f,
    delimiter=",",
)
np.savetxt(
    "thm_1_testing_loss_entropy_correlation_winners_agree_1f.csv",
    loss_entropy_correlation_winners_agree_1f,
    delimiter=",",
)
np.savetxt(
    "thm_1_testing_loss_values_1f.csv", loss_values_1f, delimiter=",",
)
np.savetxt(
    "thm_1_testing_entropy_values_1f.csv", entropy_values_1f, delimiter=",",
)
np.savetxt(
    "thm_1_testing_correlation_values_1f.csv", correlation_values_1f, delimiter=",",
)
# np.savetxt(
#     "thm_1_testing_loss_entropy_agree_2f.csv", loss_entropy_agree_2f, delimiter=","
# )
# np.savetxt(
#     "thm_1_testing_loss_importance_persists.csv",
#     loss_importance_persists,
#     delimiter=",",
# )
# np.savetxt(
#     "thm_1_testing_entropy_importance_persists.csv",
#     entropy_importance_persists,
#     delimiter=",",
# )

print(
    f"Loss/Entropy/Correlation Agreement (1 Feature): {np.mean(loss_entropy_correlation_agree_1f):.3f}"
)
print(
    f"Loss/Entropy/Correlation Winner Agreement (1 Feature): {np.mean(loss_entropy_correlation_winners_agree_1f):.3f}"
)
# print(f"Loss/Entropy Agreement (2 Features): {np.mean(loss_entropy_agree_2f):.3f}")
# print(f"Loss Importance Persists: {np.mean(loss_importance_persists):.3f}")
# print(f"Entropy Importance Persists: {np.mean(entropy_importance_persists):.3f}")

