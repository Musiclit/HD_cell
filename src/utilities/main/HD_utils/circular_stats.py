'''
functions to deal with circular data. cdiff, mean and the final
three rerange functions are the mostly commonly used.

Functions partly copied from a public but not maintained package.
The final three rerange functions are coded by myself

Siyuan Mei (mei@bio.lmu.de)
2024
'''
import warnings
import numpy as np


def cdiff(alpha, beta):
    """
    Difference between pairs :math:`x_i-y_i` around the circle,
    computed efficiently.

    :param alpha:  sample of circular random variable
    :param beta:   sample of circular random variable
    :return: distance between the pairs
    """
    return np.angle(np.exp(1j * alpha) / np.exp(1j * beta))

    
def mean(alpha, w=None, ci=None, d=None, axis=None, axial_correction=1):
    """
    Compute mean direction of circular data.

    :param alpha: circular data
    :param w: 	 weightings in case of binned angle data
    :param ci: if not None, the upper and lower 100*ci% confidence
               interval is returned as well
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :return: circular mean if ci=None, or circular mean as well as lower and
             upper confidence interval limits

    Example:   ### TODO: fix this example. Imports are not clear ###

    >>> import numpy as np
    >>> data = 2*np.pi*np.random.rand(10)
    >>> mu, (ci_l, ci_u) = mean(data, ci=0.95)

    """

    cmean = _complex_mean(alpha,
                          w=w,
                          axis=axis,
                          axial_correction=axial_correction)

    mu = np.angle(cmean) / axial_correction

    if ci is None:
        return mu
    # else:
    #     if axial_correction > 1:  # TODO: implement CI for axial correction
    #         warnings.warn("Axial correction ignored for confidence intervals.")
    #     t = mean_ci_limits(alpha, ci=ci, w=w, d=d, axis=axis)
    #     return mu, CI(mu - t, mu + t)

def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
        str(w.shape) + " do not match!"

    return ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) /
            np.sum(w, axis=axis))

def moment(alpha, p=1, cent=False,
           w=None, d=None, axis=None,
           ci=None, bootstrap_iter=None):
    """
    Computes the complex p-th centred or non-centred moment of the angular
    data in alpha.

    :param alpha: sample of angles in radian
    :param p:     the p-th moment to be computed; default is 1.
    :param cent:  if True, compute central moments. Default False.
    :param w:     number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension,
                  default is None (across all dimensions)
    :param ci: if not None, confidence level is bootstrapped
    :param bootstrap_iter: number of bootstrap iterations
                           (number of samples if None)
    :return:    the complex p-th moment.
                rho_p   magnitude of the p-th moment
                mu_p    angle of the p-th moment

    Example:

        import numpy as np
        import pycircstat as circ
        data = 2*np.pi*np.random.rand(10)
        mp = circ.moment(data)

    You can then calculate the magnitude and angle of the p-th moment:

        rho_p = np.abs(mp)  # magnitude
        mu_p = np.angle(mp)  # angle

    You can also calculate bootstrap confidence intervals:

        mp, (ci_l, ci_u) = circ.moment(data, ci=0.95)

    References: [Fisher1995]_ p. 33/34
    """

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    if cent:
        theta = mean(alpha, w=w, d=d, axis=axis)
        theta2 = np.tile(theta, (alpha.shape[0],) + len(theta.shape) * (1,))
        alpha = cdiff(alpha, theta2)

    n = alpha.shape[axis]
    cbar = np.sum(np.cos(p * alpha) * w, axis) / n
    sbar = np.sum(np.sin(p * alpha) * w, axis) / n
    mp = cbar + 1j * sbar

    return mp

def resultant_vector_length(alpha, w=None, d=None, axis=None,
                            axial_correction=1, ci=None, bootstrap_iter=None):
    """
    Computes mean resultant vector length for circular data.

    This statistic is sometimes also called vector strength.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain length
    r = np.abs(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    return r

# defines synonym for resultant_vector_length
vector_strength = resultant_vector_length

def skewness(
        alpha,
        w=None,
        axis=None,
        ci=None,
        bootstrap_iter=None,
        mode='pewsey'):
    """
    Calculates a measure of angular skewness.

    :param alpha:       sample of angles
    :param w:           weightings in case of binned angle data
    :param axis:        statistic computed along this dimension (default None, collapse dimensions)
    :param ci:          if not None, confidence level is bootstrapped
    :param bootstrap_iter: number of bootstrap iterations
    :param mode:        which skewness to compute (options are 'pewsey' or 'fisher'; 'pewsey' is default)
    :return:            the skewness
    :raise ValueError:

    References: [Pewsey2004]_, [Fisher1995]_ p. 34
    """
    if w is None:
        w = np.ones_like(alpha)
    else:
        assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    # compute neccessary values
    theta = mean(alpha, w=w, axis=axis)

    # compute skewness
    if mode == 'pewsey':
        theta2 = np.tile(theta, (alpha.shape[0],) + len(theta.shape) * (1,))
        return np.sum(
            w * np.sin(2 * cdiff(alpha, theta2)), axis=axis) / np.sum(w, axis=axis)
    elif mode == 'fisher':
        mom = moment(alpha, p=2, w=w, axis=axis, cent=True)
        mu2, rho2 = np.angle(mom), np.abs(mom)
        R = resultant_vector_length(alpha, w=w, axis=axis)
        return rho2 * np.sin(cdiff(mu2, 2 * theta)) / \
            (1 - R)**(3. / 2)  # (formula 2.29)
    else:
        raise ValueError("Mode %s not known!" % (mode, ))

def rerange(theta): # rerange theta to [-pi, pi)
    theta = theta.copy()
    if np.all( (theta < np.pi) & (theta >= -np.pi)):
        return theta
    theta[theta>=np.pi] -= 2*np.pi
    theta[theta<-np.pi] += 2*np.pi
    return rerange(theta)

def rerange_02pi(theta): # rerange theta to [0, 2pi)
    theta = theta.copy()
    if np.all( (theta < 2 * np.pi) & (theta >= 0)):
        return theta
    theta[theta>= 2*np.pi] -= 2*np.pi
    theta[theta<0] += 2*np.pi
    return rerange_02pi(theta)

def rerange_expand(arr): # expand to [-inf, inf]
    arr2 = arr.copy()
    diff = np.diff(arr2)
    jump_id1 = np.where(diff > np.pi)[0]+1
    jump_id2 = np.where(diff < -np.pi)[0]+1
    if (jump_id1.size == 0) & (jump_id2.size == 0):
        return arr2
    else:
        for id1 in jump_id1:
            arr2[id1:] = arr2[id1:] - 2*np.pi
        for id2 in jump_id2:
            arr2[id2:] = arr2[id2:] + 2*np.pi
        return rerange_expand(arr2)