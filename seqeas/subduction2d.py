"""
Module containing the rheologies, fault setup, and ODE cycles code
for the 2D subduction case.
"""

# general imports
import json
import configparser
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from numba import njit, objmode, float64, int64, boolean
from scipy.interpolate import interp1d
from warnings import warn
from abc import ABC

# seqeas imports
from .kernels2d import Glinedisp, Klinedisp


class Rheology(ABC):
    """
    Abstract base class for rheologies.
    """


class NonlinearViscous(Rheology):
    r"""
    Implement a nonlinear viscous fault rheology, where the velocity :math:`v` is
    :math:`v = \tau^n / \alpha_n` given the shear stress :math:`\tau`, a strength
    constant :math:`\alpha_n`, and a constant exponent :math:`n`.
    """

    def __init__(self, n, alpha_n, n_mid=None, alpha_n_mid=None, mid_transition=None,
                 n_deep=None, alpha_n_deep=None, deep_transition=None,
                 deep_transition_width=None, n_boundary=None, alpha_n_boundary=None):
        r"""
        Setup the rheology parameters for a given fault.

        Parameters
        ----------
        alpha_n : float
            Nonlinear viscous rheology strength constant :math:`\alpha_n` [Pa^n * s/m]
        n : float
            Power-law exponent :math:`n` [-]
        """
        # input check
        assert not np.logical_xor(deep_transition is None, deep_transition_width is None)
        # set number of variables
        self.n_vars = 2
        """ Number of variables to track by rheology [-] """
        # initialization
        self._n = float(n)
        self._n_mid = float(n_mid) if n_mid is not None else self.n
        self._n_deep = float(n_deep) if n_deep is not None else self.n_mid
        self.n_boundary = float(n_boundary) if n_boundary is not None else self.n_deep
        """ Power-law exponent :math:`n` [-] """
        self.alpha_n = float(alpha_n)
        self.alpha_n_mid = (float(alpha_n_mid) if alpha_n_mid is not None
                            else self.alpha_n)
        self.alpha_n_deep = (float(alpha_n_deep) if alpha_n_deep is not None
                             else self.alpha_n_mid)
        self.alpha_n_boundary = (float(alpha_n_boundary) if alpha_n_boundary is not None
                                 else self.alpha_n_deep)
        r""" Nonlinear viscous rheology strength constant :math:`\alpha_n` [Pa^n * s/m] """
        self.mid_transition = None if mid_transition is None else float(mid_transition)
        """ Depth [m] for the middle transition point """
        self.deep_transition = None if deep_transition is None else float(deep_transition)
        """ (Upper) Depth [m] for the deep transition point """
        self.deep_transition_width = (None if deep_transition_width is None
                                      else float(deep_transition_width))
        """ (Downdip) Width [m] of the deep transition point """

    @property
    def alpha_n(self):
        r""" Nonlinear viscous rheology strength constant :math:`\alpha_n` [Pa^n * s/m] """
        return self._alpha_n

    @alpha_n.setter
    def alpha_n(self, alpha_n):
        self._alpha_n = float(alpha_n)
        self._A = self.calc_A(self._alpha_n, self._n)

    @property
    def alpha_n_mid(self):
        r""" Nonlinear viscous rheology strength constant :math:`\alpha_n` [Pa^n * s/m] """
        return self._alpha_n_mid

    @alpha_n_mid.setter
    def alpha_n_mid(self, alpha_n_mid):
        self._alpha_n_mid = float(alpha_n_mid)
        self._A_mid = self.calc_A(self._alpha_n_mid, self._n_mid)

    @property
    def alpha_n_deep(self):
        r""" Nonlinear viscous rheology strength constant :math:`\alpha_n` [Pa^n * s/m] """
        return self._alpha_n_deep

    @alpha_n_deep.setter
    def alpha_n_deep(self, alpha_n_deep):
        self._alpha_n_deep = float(alpha_n_deep)
        self._A_deep = self.calc_A(self._alpha_n_deep, self._n_deep)

    @property
    def n(self):
        """ Power-law exponent :math:`n` [-] """
        return self._n

    @n.setter
    def n(self, n):
        self._n = float(n)
        self._A = self.calc_A(self._alpha_n, self._n)

    @property
    def n_mid(self):
        """ Power-law exponent :math:`n` [-] """
        return self._n_mid

    @n_mid.setter
    def n_mid(self, n_mid):
        self._n_mid = float(n_mid)
        self._A_mid = self.calc_A(self._alpha_n_mid, self._n_mid)

    @property
    def n_deep(self):
        """ Power-law exponent :math:`n` [-] """
        return self._n_deep

    @n_deep.setter
    def n_deep(self, n_deep):
        self._n_deep = float(n_deep)
        self._A_deep = self.calc_A(self._alpha_n_deep, self._n_deep)

    @property
    def A(self):
        r""" Rescaled strength term :math:`A = \alpha_n^{1/n}` [Pa * (s/m)^(1/n)] """
        return self._A

    @property
    def A_mid(self):
        r""" Rescaled strength term :math:`A = \alpha_n^{1/n}` [Pa * (s/m)^(1/n)] """
        return self._A_mid

    @property
    def A_deep(self):
        r""" Rescaled strength term :math:`A = \alpha_n^{1/n}` [Pa * (s/m)^(1/n)] """
        return self._A_deep

    @staticmethod
    def calc_A(alpha_n, n):
        """ Calculate A from alpha_n and n """
        return alpha_n ** (1 / n)

    def get_param_vectors(self, patch_depths, v_eff):
        r"""
        Calculate the depth-dependent arrays of :math:`\alpha_n`, :math:`n`, and :math:`A`,
        assuming :math:`\alpha_n` and :math:`\alpha_{n,eff}` vary log-linearly with depth,
        and :math:`n` adapts between the transition points.
        """
        assert np.all(np.diff(patch_depths) >= 0)
        # start knots list
        knots = [patch_depths[0]]
        vals_alpha_n = [self.alpha_n]
        vals_n = [self.n]
        # add optional mid transition
        if self.mid_transition is not None:
            knots.append(patch_depths[np.argmin(np.abs(patch_depths - self.mid_transition))])
            vals_alpha_n.append(self.alpha_n_mid)
            vals_n.append(self.n_mid)
        # add optional deep transition
        if self.deep_transition is not None:
            knots.append(patch_depths[np.argmin(np.abs(patch_depths - self.deep_transition))])
            vals_alpha_n.append(self.alpha_n_deep)
            vals_n.append(self.n_deep)
            knots.append(patch_depths[np.argmin(np.abs(patch_depths
                                                       - self.deep_transition
                                                       - self.deep_transition_width))])
            vals_alpha_n.append(self.alpha_n_boundary)
            vals_n.append(self.n_boundary)
        # add final value
        knots.append(patch_depths[-1])
        vals_alpha_n.append(self.alpha_n_boundary)
        vals_alpha_n = np.array(vals_alpha_n)
        vals_n.append(self.n_boundary)
        vals_n = np.array(vals_n)
        vals_alpha_eff = SubductionSimulation.get_alpha_eff(vals_alpha_n, vals_n, v_eff)
        # interpolate alpha_n and alpha_eff
        alpha_n_vec = 10**interp1d(knots, np.log10(vals_alpha_n))(patch_depths)
        alpha_eff_vec = 10**interp1d(knots, np.log10(vals_alpha_eff))(patch_depths)
        # get n and A
        n_vec = SubductionSimulation.get_n(alpha_n_vec, alpha_eff_vec, v_eff)
        A_vec = alpha_n_vec ** (1 / n_vec)
        return alpha_n_vec, n_vec, A_vec


class RateStateSteadyLogarithmic(Rheology):
    r"""
    Implement a steady-state rate-and-state rheology using the ageing law (effectively
    becoming a rate-dependent rheology) with velocity in logarithmic space defined by

    :math:`f_{ss} = f_0 + (a - b) * \zeta = \tau / \sigma_E`

    where :math:`f_{ss}` is the steady-state friction, :math:`f_0` is a reference
    friction, :math:`a` and :math:`b` are the rate-and-state frictional parameters,
    :math:`\zeta = \log (v / v_0)` is the logarithmic velocity, :math:`\tau` is the shear
    stress, and :math:`\sigma_E` is the effective fault normal stress.
    """

    def __init__(self, v_0, alpha_h, alpha_h_mid=None, mid_transition=None,
                 alpha_h_deep=None, deep_transition=None, deep_transition_width=None,
                 alpha_h_boundary=None):
        r"""
        Setup the rheology parameters for a given fault.

        Parameters
        ----------
        v_0 : float
            Reference velocity [m/s] used for the transformation into logarithmic space.
        alpha_h : float
            Rate-and-state parameter :math:`(a - b) * \sigma_E`,
            where :math:`a` and :math:`b` [-] are the rate-and-state frictional properties,
            and :math:`\sigma_E` [Pa] is effective fault normal stress.
        """
        self.alpha_h = float(alpha_h)
        r""" Rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        # input check
        assert not np.logical_xor(deep_transition is None, deep_transition_width is None)
        assert float(v_0) > 0, "RateStateSteadyLogarithmic needs to have positive v_0."
        # set number of variables
        self.n_vars = 2
        """ Number of variables to track by rheology [-] """
        # initialization
        self.v_0 = float(v_0)
        """ Reference velocity :math:`v_0` [m/s] """
        self.alpha_h = float(alpha_h)
        r""" Rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        self.alpha_h_mid = (float(alpha_h_mid) if alpha_h_mid is not None
                            else self.alpha_h)
        r""" Middle rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        self.alpha_h_deep = (float(alpha_h_deep) if alpha_h_deep is not None
                             else self.alpha_h_mid)
        r""" Deep rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        self.alpha_h_boundary = (float(alpha_h_boundary) if alpha_h_boundary is not None
                                 else self.alpha_h_deep)
        r""" Boundary-layer rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        self.mid_transition = None if mid_transition is None else float(mid_transition)
        """ Depth [m] for the middle transition point """
        self.deep_transition = None if deep_transition is None else float(deep_transition)
        """ (Upper) Depth [m] for the deep transition point """
        self.deep_transition_width = (None if deep_transition_width is None
                                      else float(deep_transition_width))
        """ (Downdip) Width [m] of the deep transition point """

    def get_param_vectors(self, patch_depths):
        r"""
        Calculate the depth-dependent array of :math:`\alpha_h`, assuming it
        varies log-linearly with depth.
        """
        assert np.all(np.diff(patch_depths) >= 0)
        # start knots list
        knots = [patch_depths[0]]
        vals_alpha_h = [self.alpha_h]
        # add optional mid transition
        if self.mid_transition is not None:
            knots.append(patch_depths[np.argmin(np.abs(patch_depths - self.mid_transition))])
            vals_alpha_h.append(self.alpha_h_mid)
        # add optional deep transition
        if self.deep_transition is not None:
            knots.append(patch_depths[np.argmin(np.abs(patch_depths - self.deep_transition))])
            vals_alpha_h.append(self.alpha_h_deep)
            knots.append(patch_depths[np.argmin(np.abs(patch_depths
                                                       - self.deep_transition
                                                       - self.deep_transition_width))])
            vals_alpha_h.append(self.alpha_h_boundary)
        # add final value
        knots.append(patch_depths[-1])
        vals_alpha_h.append(self.alpha_h_boundary)
        vals_alpha_h = np.array(vals_alpha_h)
        # interpolate alpha_n and alpha_eff
        alpha_h_vec = 10**interp1d(knots, np.log10(vals_alpha_h))(patch_depths)
        return alpha_h_vec


@njit(float64[:](float64[:], float64[:], float64[:], float64[:]), cache=True)
def dvdt_plvis(dtaudt, v, A, n):
    r"""
    Calculate the velocity derivative for a power-law viscous rheology.

    From :math:`v = \tau^n / \alpha_n` we get:

    :math:`\frac{dv}{dt} = \frac{n}{\alpha_n} \tau^{n-1} \frac{d \tau}{dt}`

    where

    :math:`\tau^{n-1} = \left( \alpha_n v \right)^{\frac{n-1}{n}}`

    simplifying to

    :math:`\frac{dv}{dt} = \frac{n}{A} v^{1-\frac{1}{n}} \frac{d \tau}{dt}`

    Parameters
    ----------
    dtaudt : numpy.ndarray
        1D array of the shear stress derivative
    v : numpy.ndarray
        1D array of the current velocity
    A : numpy.ndarray
        Rescaled nonlinear viscous rheology strength constant
    n : numpy.ndarray
        Power-law exponent

    Returns
    -------
    dvdt : numpy.ndarray
        1D array of the velocity derivative.
    """
    signs = np.sign(v)
    return (n / A) * (signs * v)**(1 - 1 / n) * dtaudt


@njit(float64[:](float64[:], float64[:]), cache=True)
def dzetadt_rdlog(dtaudt, alpha_h_vec):
    r"""
    Return the velocity derivative in logarithmic space given the current traction
    rate in linear space.

    Taking the derivative of the steady-state friction gives an explicit
    formulation for the slip acceleration :math:`\frac{d \zeta}{dt}`:

    :math:`\frac{df_{ss}}{dt} = (a-b) \frac{d \zeta}{dt}`

    Recognizing that :math:`\tau = f_{ss} \sigma_E` and assuming
    constant effective normal stress leads to
    :math:`\frac{d \tau}{dt} = \sigma_E \frac{df_{ss}}{dt}`, which
    can be rearranged to give the final expression

    :math:`\frac{d \zeta}{dt} = \frac{1}{(a-b) \sigma_E} \frac{d \tau}{dt}`

    Parameters
    ----------
    dtaudt : numpy.ndarray
        Traction derivative :math:`\frac{d \tau}{dt}` [Pa/s] in linear space
    alpha_h_vec : float
        Rate-and-state parameter :math:`(a - b) * \sigma_E`

    Returns
    -------
    dzetadt : numpy.ndarray
        Velocity derivative :math:`\frac{d \zeta}{dt}` [1/s] in logarithmic space.
    """
    return dtaudt / alpha_h_vec


@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True)
def get_new_vel_plvis(v_minus, delta_tau, alpha_n, n, A):
    r"""
    Calculate the instantaneous velocity change due to an instantaneous stress change
    to the fault patches. It is derived from:

    :math:`\tau_{+} = \tau_{-} + \Delta \tau`

    and plugging in the relationship :math:`v = \tau^n / \alpha_n`, we get

    :math:`\sqrt[n]{\alpha_n v_{+}} = \sqrt[n]{\alpha_n v_{-}} + \Delta \tau`

    and finally

    :math:`v_{+} = \frac{\left( A \sqrt[n]{v_{-}} + \Delta \tau \right)^n}{\alpha_n}`

    Parameters
    ----------
    v_minus : numpy.ndarray
        Initial velocity :math:`v_{-}` [m/s]
    delta_tau : numpy.ndarray
        Traction stress change :math:`\Delta \tau` [Pa]
    alpha_n : numpy.ndarray
        Nonlinear viscous rheology strength constant :math:`\alpha_n` [Pa^n * s/m]
    n : numpy.ndarray
        Power-law exponent :math:`n` [-]
    A : numpy.ndarray
        Rescaled strength term :math:`A = \alpha_n^{1/n}` [Pa * (s/m)^(1/n)]

    Returns
    -------
    v_plus : numpy.ndarray
        Velocity :math:`v_{+}` [m/s] after stress change
    """
    signs = np.sign(v_minus)
    temp = A * (signs * v_minus)**(1 / n) + (signs * delta_tau)
    return np.abs(temp) ** (n - 1) * temp / alpha_n * signs


@njit(float64[:](float64[:], float64[:], float64[:]), cache=True)
def get_new_vel_rdlog(zeta_minus, delta_tau, alpha_h_vec):
    r"""
    Calculate the instantaneous velocity change (in logarithmic space) due to an
    instantaneous stress change to the fault patches. We can kickstart the
    derivatuion from the expression in ``RateStateSteadyLinear.get_new_vel``:

    :math:`\log (v_{+}/v_0) = \log (v_{-}/v_0) + \Delta\tau / \alpha_h`

    and realize that we only have to plug in our definition for :math:`\zeta`
    to give us the final result

    :math:`\zeta_{+} = \zeta_{-} + \Delta\tau / \alpha_h`

    Parameters
    ----------
    zeta_minus : numpy.ndarray
        Initial velocity :math:`\zeta_{-}` [-] in logarithmic space
    delta_tau : numpy.ndarray, optional
        Traction stress change :math:`\Delta \tau` [Pa]
    alpha_h_vec : numpy.ndarray
        Rate-and-state parameter :math:`(a - b) * \sigma_E`

    Returns
    -------
    zeta_plus : numpy.ndarray
        Velocity :math:`\zeta_{+}` [-] in logarithmic space after stress change

    See Also
    --------
    alpha_h
    """
    return zeta_minus + delta_tau / alpha_h_vec


@njit(float64[:](float64, float64[:], float64, float64[:, ::1], float64[:, ::1],
                 float64[:], float64[:], float64), cache=True)
def flat_ode_plvis(t, state, v_plate, K_int, K_ext, A_upper, n_upper, mu_over_2vs):
    r"""
    Flattened ODE derivative function for a subduction fault with
    powerlaw-viscous rheology in the upper plate interface, and an imposed
    constant plate velocity at the lower interface (which can be ignored).

    Parameters
    ----------
    t : float
        Current time (needs to be in function call for solve_ivp).
    state : numpy.ndarray
        1D array with the current state of the creeping fault patches,
        containing (in order) the upper cumulative slip and upper velocity.
    v_plate : float
        Plate velocity.
    K_int : numpy.ndarray
        2D array with the stress kernel mapping creeping patches to themselves.
    K_ext : numpy.ndarray
        2D array with the stress kernel mapping the effect of the locked
        patches onto the creeping patches.
    A_upper : numpy.ndarray
        Upper plate interface rescaled nonlinear viscous rheology strength constant
    n_upper : numpy.ndarray
        Upper plate interface power-law exponent
    mu_over_2vs : float
        Radiation damping factor

    Returns
    -------
    dstatedt : numpy.ndarray
        1D array with the state derivative.
    """
    # get number of variables within state
    # (depends on rheology, so is hardcoded here)
    n_vars_upper = 2
    n_creeping_upper = state.size // n_vars_upper
    assert K_int.shape == (n_creeping_upper, n_creeping_upper)
    assert K_ext.shape[0] == n_creeping_upper
    # extract total velocities
    v = state[n_creeping_upper:]
    # get shear strain rate
    signs = np.sign(v)
    temp = mu_over_2vs * (n_upper / A_upper) * (signs * v)**(1 - 1 / n_upper)
    dtaudt = (K_int @ (v - v_plate) - np.sum(K_ext * v_plate, axis=1)
              ) / (1 + temp)
    # get ODE
    dstatedt = np.concatenate((v, dvdt_plvis(dtaudt, v, A_upper, n_upper)))
    # return
    return dstatedt


@njit(float64[:](float64, float64[:], float64, float64[:, ::1], float64[:, ::1],
                 float64, float64[:], float64), cache=True)
def flat_ode_rdlog(t, state, v_plate, K_int, K_ext, v_0, alpha_h_vec, mu_over_2vs):
    r"""
    Flattened ODE derivative function for a subduction fault with
    powerlaw-viscous rheology in the upper plate interface, and an imposed
    constant plate velocity at the lower interface (which can be ignored).

    Parameters
    ----------
    t : float
        Current time (needs to be in function call for solve_ivp).
    state : numpy.ndarray
        1D array with the current state of the creeping fault patches,
        containing (in order) the upper cumulative slip and upper velocity.
    v_plate : float
        Plate velocity.
    K_int : numpy.ndarray
        2D array with the stress kernel mapping creeping patches to themselves.
    K_ext : numpy.ndarray
        2D array with the stress kernel mapping the effect of the locked
        patches onto the creeping patches.
    v_0 : float
        Reference velocity [m/s]
    alpha_h_vec : numpy.ndarray
        Rate-and-state parameter :math:`(a - b) * \sigma_E`
    mu_over_2vs : float
        Radiation damping factor

    Returns
    -------
    dstatedt : numpy.ndarray
        1D array with the state derivative.
    """
    # get number of variables within state
    # (depends on rheology, so is hardcoded here)
    n_vars_upper = 2
    n_creeping_upper = state.size // n_vars_upper
    assert K_int.shape == (n_creeping_upper, n_creeping_upper)
    assert K_ext.shape[0] == n_creeping_upper
    # extract total velocities
    zeta = state[n_creeping_upper:]
    v = v_0 * np.exp(zeta)
    # get shear strain rate
    temp = mu_over_2vs * v / alpha_h_vec
    dtaudt = (K_int @ (v - v_plate) - np.sum(K_ext * v_plate, axis=1)
              ) / (1 + temp)
    # get ODE
    dstatedt = np.concatenate((v, dzetadt_rdlog(dtaudt, alpha_h_vec)))
    # return
    return dstatedt


@njit(float64[:](float64, float64[:], int64, float64[:], float64[:, ::1], float64[:, ::1],
                 float64, float64, float64, float64), cache=True)
def flat_ode_plvis_plvis(t, state, n_creeping_upper, v_plate_vec, K_int, K_ext,
                         A_upper, n_upper, A_lower, n_lower):
    """
    Flattened ODE derivative function for a subduction fault with
    powerlaw-viscous rheology in both the upper and lower plate interface.

    Parameters
    ----------
    t : float
        Current time (needs to be in function call for solve_ivp).
    state : numpy.ndarray
        1D array with the current state of the creeping fault patches,
        containing (in order) the upper cumulative slip, upper velocity,
        lower cumulative slip, lower velocity.
    n_creeping_upper : int
        Number of creeping patches in the upper plate interface.
        The number of creeping patches in the lower plate interface can then
        be derived from the size of ``state``.
    v_plate_vec : float
        Initial velocity in all creeping patches.
    K_int : numpy.ndarray
        2D array with the stress kernel mapping creeping patches to themselves.
    K_ext : numpy.ndarray
        2D array with the stress kernel mapping the effect of the locked
        patches onto the creeping patches.
    A_upper : float
        Upper plate interface rescaled nonlinear viscous rheology strength constant
    n_upper : float
        Upper plate interface power-law exponent
    A_lower : float
        Lower plate interface rescaled nonlinear viscous rheology strength constant
    n_lower : float
        Lower plate interface power-law exponent

    Returns
    -------
    dstatedt : numpy.ndarray
        1D array with the state derivative.
    """
    # get number of variables within state
    # (depends on rheology, so is hardcoded here)
    n_vars_upper, n_vars_lower = 2, 2
    n_state_upper = n_vars_upper * n_creeping_upper
    n_state_lower = state.size - n_state_upper
    n_creeping_lower = n_state_lower // n_vars_lower
    n_creeping = n_creeping_lower + n_creeping_upper
    assert K_int.shape[0] == K_int.shape[1] == n_creeping
    assert K_ext.shape[0] == n_creeping
    # split up state
    state_upper = state[:n_state_upper]
    state_lower = state[n_state_upper:]
    # extract total velocities
    v_upper = state_upper[n_creeping_upper:]
    v_lower = state_lower[n_creeping_lower:]
    # get shear strain rate
    v = np.concatenate((v_upper, v_lower))
    dtaudt = (K_int @ (v - v_plate_vec) - np.sum(K_ext * v_plate_vec[0], axis=1))
    dtaudt_upper = dtaudt[:n_creeping_upper]
    dtaudt_lower = dtaudt[n_creeping_upper:]
    # get individual rheologies' ODE
    dstatedt_upper = \
        np.concatenate((v_upper, dvdt_plvis(dtaudt_upper, v_upper,
                                            np.ones_like(v_upper) * A_upper,
                                            np.ones_like(v_upper) * n_upper)))
    dstatedt_lower = \
        np.concatenate((v_lower, dvdt_plvis(dtaudt_lower, v_lower,
                                            np.ones_like(v_lower) * A_lower,
                                            np.ones_like(v_upper) * n_lower)))
    # concatenate and return
    return np.concatenate((dstatedt_upper, dstatedt_lower))


@njit(float64[:](float64, float64[:], int64, float64[:], float64[:, ::1], float64[:, ::1],
                 float64, float64, float64, float64), cache=True)
def flat_ode_rdlog_plvis(t, state, n_creeping_upper, v_plate_vec, K_int, K_ext,
                         v_0, alpha_h_upper, A_lower, n_lower):
    r"""
    Flattened ODE derivative function for a subduction fault with
    rate-dependent (log-space) rheology in the upper and nonlinear viscous
    rheology in the lower plate interface.

    Parameters
    ----------
    t : float
        Current time (needs to be in function call for solve_ivp).
    state : numpy.ndarray
        1D array with the current state of the creeping fault patches,
        containing (in order) the upper cumulative slip, upper velocity,
        lower cumulative slip, lower velocity.
    n_creeping_upper : int
        Number of creeping patches in the upper plate interface.
        The number of creeping patches in the lower plate interface can then
        be derived from the size of ``state``.
    v_plate_vec : float
        Initial velocity in all creeping patches.
    K_int : numpy.ndarray
        2D array with the stress kernel mapping creeping patches to themselves.
    K_ext : numpy.ndarray
        2D array with the stress kernel mapping the effect of the locked
        patches onto the creeping patches.
    v_0 : float
        Reference velocity [m/s]
    alpha_h_upper : float
        Upper interface rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa]
    A_lower : float
        Lower plate interface rescaled nonlinear viscous rheology strength constant
    n_lower : float
        Lower plate interface power-law exponent

    Returns
    -------
    dstatedt : numpy.ndarray
        1D array with the state derivative.
    """
    # get number of variables within state
    # (depends on rheology, so is hardcoded here)
    n_vars_upper, n_vars_lower = 2, 2
    n_state_upper = n_vars_upper * n_creeping_upper
    n_state_lower = state.size - n_state_upper
    n_creeping_lower = n_state_lower // n_vars_lower
    n_creeping = n_creeping_lower + n_creeping_upper
    assert K_int.shape[0] == K_int.shape[1] == n_creeping
    assert K_ext.shape[0] == n_creeping
    # split up state
    state_upper = state[:n_state_upper]
    state_lower = state[n_state_upper:]
    # extract total velocities
    v_upper = v_0 * np.exp(state_upper[n_creeping_upper:])
    v_lower = state_lower[n_creeping_lower:]
    # get shear strain rate
    v = np.concatenate((v_upper, v_lower))
    dtaudt = (K_int @ (v - v_plate_vec) - np.sum(K_ext * v_plate_vec[0], axis=1))
    dtaudt_upper = dtaudt[:n_creeping_upper]
    dtaudt_lower = dtaudt[n_creeping_upper:]
    # get individual rheologies' ODE
    dstatedt_upper = \
        np.concatenate((v_upper, dzetadt_rdlog(dtaudt_upper,
                                               np.ones_like(v_lower) * alpha_h_upper)))
    dstatedt_lower = \
        np.concatenate((v_lower, dvdt_plvis(dtaudt_lower, v_lower,
                                            np.ones_like(v_lower) * A_lower,
                                            np.ones_like(v_upper) * n_lower)))
    # concatenate and return
    return np.concatenate((dstatedt_upper, dstatedt_lower))


# simple rk4
@njit(float64[:, :](float64, float64, float64[:], float64[:], int64, float64[:],
                    float64[:, ::1], float64[:, ::1], float64, float64, float64, float64),
      cache=True)
def myrk4(ti, tf, state0, t_eval, n_creeping_upper, v_plate_vec,
          K_int, K_ext, A_upper, n_upper, A_lower, n_lower):
    h = t_eval[1] - t_eval[0]
    num_state = state0.size
    num_eval = t_eval.size
    sol = np.zeros((num_eval, num_state))
    sol[0, :] = state0
    for i in range(1, num_eval):
        cur = sol[i-1, :]
        k1 = flat_ode_plvis_plvis(ti, cur, n_creeping_upper, v_plate_vec, K_int, K_ext,
                                  A_upper, n_upper, A_lower, n_lower)
        cur = sol[i-1, :] + (h / 2) * k1
        k2 = flat_ode_plvis_plvis(ti, cur, n_creeping_upper, v_plate_vec, K_int, K_ext,
                                  A_upper, n_upper, A_lower, n_lower)
        cur = sol[i-1, :] + (h / 2) * k2
        k3 = flat_ode_plvis_plvis(ti, cur, n_creeping_upper, v_plate_vec, K_int, K_ext,
                                  A_upper, n_upper, A_lower, n_lower)
        cur = sol[i-1, :] + h * k3
        k4 = flat_ode_plvis_plvis(ti, cur, n_creeping_upper, v_plate_vec, K_int, K_ext,
                                  A_upper, n_upper, A_lower, n_lower)
        sol[i, :] = sol[i-1, :] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return sol


@njit(float64[:, :](float64[:], int64[:], int64[:], int64, int64, float64[:, ::1],
                    float64[:, ::1], float64[:], float64[:], float64[:, ::1], float64[:, ::1],
                    float64[:], float64[:], float64[:], float64), cache=True)
def flat_run_plvis(t_eval, i_break, i_eq,
                   n_creeping_upper, n_creeping_lower, K_int, K_ext,
                   v_plate_vec, v_init, slip_taper, delta_tau_bounded,
                   alpha_n_vec, n_vec, A_vec, mu_over_2vs):
    r"""
    Run the simulation.

    Parameters
    ----------
    t_eval : numpy.ndarray
        Evaluation times [s]
    i_break : numpy.ndarray
        Integer indices of cycle breaks [-]
    i_eq : numpy.ndarray
        Integer indices of earthquakes within sequence [-]
    n_creeping_upper : int
        Number [-] of creeping patches in the upper fault interface
    n_creeping_lower : int
        Number [-] of creeping patches in the lower fault interface
    K_int : numpy.ndarray
        Internal stress kernel [Pa/m]
    K_ext : numpy.ndarray
        External stress kernel [Pa/m]
    v_plate_vec : numpy.ndarray
        Plate velocity for all creeping patches [m/s]
    v_init : numpy.ndarray
        Initial velocity of the fault patches, in the dimensions of the rheology
    slip_taper : numpy.ndarray
        Compensating coseismic tapered slip on creeping patches [m]
    delta_tau_bounded : numpy.ndarray
        Bounded coseismic stress change [Pa]
    alpha_n_vec : numpy.ndarray
        Upper plate interface nonlinear viscous rheology strength constant [Pa^n * s/m]
        at each patch
    n_vec : float
        Upper plate interface power-law exponent [-] at each patch
    A_vec : float
        Rescaled upper plate interface nonlinear viscous rheology strength constant
        [Pa^n * s/m] at each patch
    mu_over_2vs : float
        Radiation damping factor :math:`\mu / 2 v_s`, where :math:`\mu` is the shear
        modulus [Pa] and :math:`v_s` is the shear wave velocity [m/s]

    Returns
    -------
    full_state : numpy.ndarray
        Full state variable at the end of the integration.
    """
    # initialize parameters
    n_vars_upper, n_vars_lower = 2, 2
    n_state_upper = n_vars_upper * n_creeping_upper
    n_state_lower = n_vars_lower * n_creeping_lower
    n_eval = t_eval.size
    n_slips = delta_tau_bounded.shape[1]

    # initialize arrays
    s_minus_upper = np.zeros((n_vars_upper - 1) * n_creeping_upper)
    s_minus_lower = np.zeros(n_creeping_lower)
    v_minus_upper = v_init[:n_creeping_upper]
    v_minus_lower = v_plate_vec[n_creeping_upper:]
    full_state = np.empty((n_state_upper + n_state_lower, n_eval))
    full_state[:] = np.NaN
    state_plus = np.concatenate((s_minus_upper, v_minus_upper, s_minus_lower, v_minus_lower))

    # make flat ODE function arguments
    args = (v_plate_vec[0], K_int[:n_creeping_upper, :n_creeping_upper].copy(),
            K_ext[:n_creeping_upper, :], A_vec, n_vec, mu_over_2vs)

    # integrate
    spun_up = 0
    i_slip = 0
    steps = np.sort(np.concatenate((i_eq, i_break)))
    i = 0
    atol = np.ones(n_state_upper) * 1e-6
    atol[n_creeping_upper:] = 1e-15
    while i < steps.size - 1:
        # print(f"{i+1}/{steps.size - 1}")
        # get indices
        ji, jf = steps[i], steps[i+1]
        ti, tf = t_eval[ji], t_eval[jf]
        # call integrator
        with objmode(sol="float64[:, :]", success="boolean"):
            sol = solve_ivp(flat_ode_plvis,
                            t_span=[ti, tf],
                            y0=state_plus[:n_state_upper],
                            t_eval=t_eval[ji:jf + 1],
                            method="LSODA", rtol=1e-6, atol=atol, args=args)
            success = sol.success
            if success:
                sol = sol.y
            else:
                sol = np.empty((1, 1))
        if not success:
            raise RuntimeError("Integrator failed.")
        # save state to output array
        full_state[:n_state_upper, ji:jf + 1] = sol
        # fill in the imposed lower state
        full_state[n_state_upper:n_state_upper + n_creeping_lower, ji:jf + 1] = \
            np.ascontiguousarray(v_plate_vec[n_creeping_upper:]).reshape((-1, 1)) \
            * np.ascontiguousarray(t_eval[ji:jf + 1]).reshape((1, -1))
        full_state[n_state_upper + n_creeping_lower:, ji:jf + 1] = \
            np.ascontiguousarray(v_plate_vec[n_creeping_upper:]).reshape((-1, 1))
        # can already stop here if this is the last interval
        if i == steps.size - 2:
            break
        # at the end of a full cycle, check the early stopping criteria
        if (not spun_up) and (i > n_slips) and (jf in i_break):
            old_full_state = full_state[:, steps[i-2*n_slips-1]:steps[i-n_slips]]
            new_full_state = full_state[:, steps[i-n_slips]:steps[i+1]]
            old_state_upper = old_full_state[:n_state_upper, :]
            new_state_upper = new_full_state[:n_state_upper, :]
            old_v_upper = old_state_upper[-n_creeping_upper:, -1]
            new_v_upper = new_state_upper[-n_creeping_upper:, -1]
            lhs_upper = np.abs(old_v_upper - new_v_upper)
            rhs_upper = (1e-3) * np.abs(v_plate_vec[0]) + (1e-3) * np.abs(new_v_upper)
            stop_now = np.all(lhs_upper <= rhs_upper)
            if stop_now:
                spun_up = jf
                # advance i to the last cycle (don't forget the general advance later)
                i = steps.size - n_slips - 3
        elif spun_up and (jf in i_break):
            break
        # apply step change only if there is one
        if (jf in i_eq):
            state_upper, state_lower = sol[:n_state_upper, -1], sol[n_state_upper:, -1]
            s_minus_upper = state_upper[:-n_creeping_upper]
            v_minus_upper = state_upper[-n_creeping_upper:]
            s_minus_lower = state_lower[:-n_creeping_lower]
            v_minus_lower = state_lower[-n_creeping_lower:]
            s_plus_upper = s_minus_upper.ravel().copy()
            s_plus_upper[:n_creeping_upper] += slip_taper[:, i_slip]
            s_plus_lower = s_minus_lower.ravel()
            v_plus_upper = get_new_vel_plvis(v_minus_upper,
                                             delta_tau_bounded[:n_creeping_upper, i_slip],
                                             alpha_n_vec, n_vec, A_vec)
            v_plus_lower = v_minus_lower.ravel()
            state_plus = np.concatenate((s_plus_upper, v_plus_upper,
                                         s_plus_lower, v_plus_lower))
            i_slip = (i_slip + 1) % n_slips
        else:
            state_plus = sol[:, -1]
        # advance
        i += 1

    # warn if we never spun up
    if not spun_up:
        print(f"Simulation did not spin up after {len(i_break) - 1} cycles!")

    # done
    return full_state


@njit(float64[:, :](float64[:], int64[:], int64[:], int64, int64, float64[:, ::1],
                    float64[:, ::1], float64[:], float64[:], float64[:, ::1], float64[:, ::1],
                    float64, float64[:], float64), cache=True)
def flat_run_rdlog(t_eval, i_break, i_eq,
                   n_creeping_upper, n_creeping_lower, K_int, K_ext,
                   v_plate_vec, v_init, slip_taper, delta_tau_bounded,
                   v_0, alpha_h_vec, mu_over_2vs):
    r"""
    Run the simulation.

    Parameters
    ----------
    t_eval : numpy.ndarray
        Evaluation times [s]
    i_break : numpy.ndarray
        Integer indices of cycle breaks [-]
    i_eq : numpy.ndarray
        Integer indices of earthquakes within sequence [-]
    n_creeping_upper : int
        Number [-] of creeping patches in the upper fault interface
    n_creeping_lower : int
        Number [-] of creeping patches in the lower fault interface
    K_int : numpy.ndarray
        Internal stress kernel [Pa/m]
    K_ext : numpy.ndarray
        External stress kernel [Pa/m]
    v_plate_vec : numpy.ndarray
        Plate velocity for all creeping patches [m/s]
    v_init : numpy.ndarray
        Initial velocity of the fault patches, in the dimensions of the rheology
    slip_taper : numpy.ndarray
        Compensating coseismic tapered slip on creeping patches [m]
    delta_tau_bounded : numpy.ndarray
        Bounded coseismic stress change [Pa]
    v_0 : float
        Reference velocity [m/s]
    alpha_h_vec : numpy.ndarray
        Upper interface rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa]
    mu_over_2vs : float
        Radiation damping factor :math:`\mu / 2 v_s`, where :math:`\mu` is the shear
        modulus [Pa] and :math:`v_s` is the shear wave velocity [m/s]

    Returns
    -------
    full_state : numpy.ndarray
        Full state variable at the end of the integration.
    """
    # initialize parameters
    n_vars_upper, n_vars_lower = 2, 2
    n_state_upper = n_vars_upper * n_creeping_upper
    n_state_lower = n_vars_lower * n_creeping_lower
    n_eval = t_eval.size
    n_slips = delta_tau_bounded.shape[1]

    # initialize arrays
    s_minus_upper = np.zeros((n_vars_upper - 1) * n_creeping_upper)
    s_minus_lower = np.zeros(n_creeping_lower)
    assert np.all(v_init[:n_creeping_upper] > 0)
    zeta_minus_upper = np.log(v_init[:n_creeping_upper] / v_0)
    v_minus_lower = v_plate_vec[n_creeping_upper:]
    full_state = np.empty((n_state_upper + n_state_lower, n_eval))
    full_state[:] = np.NaN
    state_plus = np.concatenate((s_minus_upper, zeta_minus_upper,
                                 s_minus_lower, v_minus_lower))

    # make flat ODE function arguments
    args = (v_plate_vec[0], K_int[:n_creeping_upper, :n_creeping_upper].copy(),
            K_ext[:n_creeping_upper, :], v_0, alpha_h_vec, mu_over_2vs)

    # integrate
    spun_up = 0
    i_slip = 0
    steps = np.sort(np.concatenate((i_eq, i_break)))
    i = 0
    while i < steps.size - 1:
        # print(f"{i+1}/{steps.size - 1}")
        # get indices
        ji, jf = steps[i], steps[i+1]
        ti, tf = t_eval[ji], t_eval[jf]
        # call integrator
        with objmode(sol="float64[:, :]", success="boolean"):
            sol = solve_ivp(flat_ode_rdlog,
                            t_span=[ti, tf],
                            y0=state_plus[:n_state_upper],
                            t_eval=t_eval[ji:jf + 1],
                            method="LSODA", args=args)
            success = sol.success
            if success:
                sol = sol.y
            else:
                sol = np.empty((1, 1))
        if not success:
            raise RuntimeError("Integrator failed.")
        # save state to output array
        full_state[:n_state_upper, ji:jf + 1] = sol
        # fill in the imposed lower state
        full_state[n_state_upper:n_state_upper + n_creeping_lower, ji:jf + 1] = \
            np.ascontiguousarray(v_plate_vec[n_creeping_upper:]).reshape((-1, 1)) \
            * np.ascontiguousarray(t_eval[ji:jf + 1]).reshape((1, -1))
        full_state[n_state_upper + n_creeping_lower:, ji:jf + 1] = \
            np.ascontiguousarray(v_plate_vec[n_creeping_upper:]).reshape((-1, 1))
        # can already stop here if this is the last interval
        if i == steps.size - 2:
            break
        # at the end of a full cycle, check the early stopping criteria
        if (not spun_up) and (i > n_slips) and (jf in i_break):
            old_full_state = full_state[:, steps[i-2*n_slips-1]:steps[i-n_slips]]
            new_full_state = full_state[:, steps[i-n_slips]:steps[i+1]]
            old_state_upper = old_full_state[:n_state_upper, :]
            new_state_upper = new_full_state[:n_state_upper, :]
            old_v_upper = v_0 * np.exp(old_state_upper[-n_creeping_upper:, -1])
            new_v_upper = v_0 * np.exp(new_state_upper[-n_creeping_upper:, -1])
            lhs_upper = np.abs(old_v_upper - new_v_upper)
            rhs_upper = (1e-3) * np.abs(v_plate_vec[0]) + (1e-3) * np.abs(new_v_upper)
            stop_now = np.all(lhs_upper <= rhs_upper)
            if stop_now:
                spun_up = jf
                # advance i to the last cycle (don't forget the general advance later)
                i = steps.size - n_slips - 3
        elif spun_up and (jf in i_break):
            break
        # apply step change only if there is one
        if (jf in i_eq):
            state_upper, state_lower = sol[:n_state_upper, -1], sol[n_state_upper:, -1]
            s_minus_upper = state_upper[:-n_creeping_upper]
            zeta_minus_upper = state_upper[-n_creeping_upper:]
            s_minus_lower = state_lower[:-n_creeping_lower]
            v_minus_lower = state_lower[-n_creeping_lower:]
            s_plus_upper = s_minus_upper.ravel().copy()
            s_plus_upper[:n_creeping_upper] += slip_taper[:, i_slip]
            s_plus_lower = s_minus_lower.ravel()
            zeta_plus_upper = get_new_vel_rdlog(zeta_minus_upper,
                                                delta_tau_bounded[:n_creeping_upper, i_slip],
                                                alpha_h_vec)
            v_plus_lower = v_minus_lower.ravel()
            state_plus = np.concatenate((s_plus_upper, zeta_plus_upper,
                                         s_plus_lower, v_plus_lower))
            i_slip = (i_slip + 1) % n_slips
        else:
            state_plus = sol[:, -1]
        # advance
        i += 1

    # warn if we never spun up
    if not spun_up:
        print(f"Simulation did not spin up after {len(i_break) - 1} cycles!")

    full_state[n_creeping_upper:n_state_upper, :] = \
        v_0 * np.exp(full_state[n_creeping_upper:n_state_upper, :])

    # done
    return full_state


@njit(float64[:, :](float64[:], int64[:], int64[:], int64, int64, float64[:, ::1],
                    float64[:, ::1], float64[:], float64[:], float64[:, ::1], float64[:, ::1],
                    float64, float64, float64, float64, boolean), cache=True)
def flat_run_plvis_plvis(t_eval, i_break, i_eq,
                         n_creeping_upper, n_creeping_lower, K_int, K_ext,
                         v_plate_vec, v_init, slip_taper, delta_tau_bounded,
                         alpha_n_upper, n_upper, alpha_n_lower, n_lower,
                         simple_rk4):
    """
    Run the simulation.

    Parameters
    ----------
    t_eval : numpy.ndarray
        Evaluation times [s]
    i_break : numpy.ndarray
        Integer indices of cycle breaks [-]
    i_eq : numpy.ndarray
        Integer indices of earthquakes within sequence [-]
    n_creeping_upper : int
        Number [-] of creeping patches in the upper fault interface
    n_creeping_lower : int
        Number [-] of creeping patches in the lower fault interface
    K_int : numpy.ndarray
        Internal stress kernel [Pa/m]
    K_ext : numpy.ndarray
        External stress kernel [Pa/m]
    v_plate_vec : numpy.ndarray
        Plate velocity for all creeping patches [m/s]
    v_init : numpy.ndarray
        Initial velocity of the fault patches, in the dimensions of the rheology
    slip_taper : numpy.ndarray
        Compensating coseismic tapered slip on creeping patches [m]
    delta_tau_bounded : numpy.ndarray
        Bounded coseismic stress change [Pa]
    alpha_n_upper : float
        Upper plate interface nonlinear viscous rheology strength constant  [Pa^n * s/m]
    n_upper : float
        Upper plate interface power-law exponent [-]
    alpha_n_lower : float
        Lower plate interface nonlinear viscous rheology strength constant  [Pa^n * s/m]
    n_lower : float
        Lower plate interface power-law exponent [-]
    simple_rk4 : bool
        Decide whether to use the simple RK4 integrator or not

    Returns
    -------
    full_state : numpy.ndarray
        Full state variable at the end of the integration.
    """
    # initialize parameters
    n_vars_upper, n_vars_lower = 2, 2
    n_state_upper = n_vars_upper * n_creeping_upper
    n_state_lower = n_vars_lower * n_creeping_lower
    A_upper = alpha_n_upper ** (1 / n_upper)
    A_lower = alpha_n_lower ** (1 / n_lower)
    n_eval = t_eval.size
    n_slips = delta_tau_bounded.shape[1]

    # initialize arrays
    s_minus_upper = np.zeros((n_vars_upper - 1) * n_creeping_upper)
    s_minus_lower = np.zeros(n_creeping_lower)
    v_minus_upper = v_init[:n_creeping_upper]
    # if isinstance(self.fault.upper_rheo, rheologies.RateStateSteadyLogarithmic):
    #     v_minus_upper = self.fault.upper_rheo.v2zeta(v_minus_upper)
    v_minus_lower = v_init[n_creeping_upper:]
    full_state = np.empty((n_state_upper + n_state_lower, n_eval))
    full_state[:] = np.NaN
    state_plus = np.concatenate((s_minus_upper, v_minus_upper, s_minus_lower, v_minus_lower))

    # make flat ODE function arguments
    args = (n_creeping_upper, v_plate_vec, K_int, K_ext,
            A_upper, n_upper, A_lower, n_lower)

    # integrate
    spun_up = 0
    i_slip = 0
    steps = np.sort(np.concatenate((i_eq, i_break)))
    i = 0
    while i < steps.size - 1:
        # get indices
        ji, jf = steps[i], steps[i+1]
        ti, tf = t_eval[ji], t_eval[jf]
        # call integrator
        if simple_rk4:
            sol = myrk4(ti, tf, state_plus, t_eval[ji:jf + 1], *args).T
        else:
            with objmode(sol="float64[:, :]", success="boolean"):
                sol = solve_ivp(flat_ode_plvis_plvis,
                                t_span=[ti, tf],
                                y0=state_plus,
                                t_eval=t_eval[ji:jf + 1],
                                method="RK45", rtol=1e-9, atol=1e-12, args=args)
                success = sol.success
                sol = sol.y
            if not success:
                raise RuntimeError("Integrator failed.")
        # save state to output array
        full_state[:, ji:jf + 1] = sol
        # can already stop here if this is the last interval
        if i == steps.size - 2:
            break
        # at the end of a full cycle, check the early stopping criteria
        if (not spun_up) and (i > n_slips) and (jf in i_break):
            old_full_state = full_state[:, steps[i-2*n_slips-1]:steps[i-n_slips]]
            new_full_state = full_state[:, steps[i-n_slips]:steps[i+1]]
            old_state_upper = old_full_state[:n_state_upper, :]
            old_state_lower = old_full_state[n_state_upper:, :]
            new_state_upper = new_full_state[:n_state_upper, :]
            new_state_lower = new_full_state[n_state_upper:, :]
            old_v_upper = old_state_upper[-n_creeping_upper:, -1]
            old_v_lower = old_state_lower[-n_creeping_lower:, -1]
            new_v_upper = new_state_upper[-n_creeping_upper:, -1]
            new_v_lower = new_state_lower[-n_creeping_lower:, -1]
            # if isinstance(self.fault.upper_rheo, rheologies.RateStateSteadyLogarithmic):
            #     old_v_upper = self.fault.upper_rheo.zeta2v(old_v_upper)
            #     new_v_upper = self.fault.upper_rheo.zeta2v(new_v_upper)
            lhs_upper = np.abs(old_v_upper - new_v_upper)
            lhs_lower = np.abs(old_v_lower - new_v_lower)
            rhs_upper = (1e-4) * np.abs(v_plate_vec[0]) + (1e-4) * np.abs(new_v_upper)
            rhs_lower = (1e-4) * np.abs(v_plate_vec[-1]) + (1e-4) * np.abs(new_v_lower)
            stop_now = np.all(lhs_upper <= rhs_upper) & np.all(lhs_lower <= rhs_lower)
            if stop_now:
                spun_up = jf
                # advance i to the last cycle (don't forget the general advance later)
                i = steps.size - n_slips - 3
        elif spun_up and (jf in i_break):
            break
        # apply step change only if there is one
        if (jf in i_eq):
            state_upper, state_lower = sol[:n_state_upper, -1], sol[n_state_upper:, -1]
            s_minus_upper = state_upper[:-n_creeping_upper]
            v_minus_upper = state_upper[-n_creeping_upper:]
            s_minus_lower = state_lower[:-n_creeping_lower]
            v_minus_lower = state_lower[-n_creeping_lower:]
            s_plus_upper = s_minus_upper.ravel().copy()
            s_plus_upper[:n_creeping_upper] += slip_taper[:, i_slip]
            s_plus_lower = s_minus_lower.ravel()
            v_plus_upper = get_new_vel_plvis(v_minus_upper,
                                             delta_tau_bounded[:n_creeping_upper, i_slip],
                                             np.ones(n_creeping_upper) * alpha_n_upper,
                                             np.ones(n_creeping_upper) * n_upper,
                                             np.ones(n_creeping_upper) * A_upper)
            v_plus_lower = get_new_vel_plvis(v_minus_lower,
                                             delta_tau_bounded[n_creeping_upper:, i_slip],
                                             np.ones(n_creeping_upper) * alpha_n_lower,
                                             np.ones(n_creeping_upper) * n_lower,
                                             np.ones(n_creeping_upper) * A_lower)
            state_plus = np.concatenate((s_plus_upper, v_plus_upper,
                                         s_plus_lower, v_plus_lower))
            i_slip = (i_slip + 1) % n_slips
        else:
            state_plus = sol[:, -1]
        # advance
        i += 1

    # warn if we never spun up
    if not spun_up:
        print(f"Simulation did not spin up after {len(i_break) - 1} cycles!")

    # if isinstance(self.fault.upper_rheo, rheologies.RateStateSteadyLogarithmic):
    #     vel_upper = self.fault.upper_rheo.zeta2v(vel_upper)

    # done
    return full_state


@njit(float64[:, :](float64[:], int64[:], int64[:], int64, int64, float64[:, ::1],
                    float64[:, ::1], float64[:], float64[:], float64[:, ::1], float64[:, ::1],
                    float64, float64, float64, float64, boolean), cache=True)
def flat_run_rdlog_plvis(t_eval, i_break, i_eq,
                         n_creeping_upper, n_creeping_lower, K_int, K_ext,
                         v_plate_vec, v_init, slip_taper, delta_tau_bounded,
                         v_0, alpha_h_upper, alpha_n_lower, n_lower,
                         simple_rk4):
    r"""
    Run the simulation.

    Parameters
    ----------
    t_eval : numpy.ndarray
        Evaluation times [s]
    i_break : numpy.ndarray
        Integer indices of cycle breaks [-]
    i_eq : numpy.ndarray
        Integer indices of earthquakes within sequence [-]
    n_creeping_upper : int
        Number [-] of creeping patches in the upper fault interface
    n_creeping_lower : int
        Number [-] of creeping patches in the lower fault interface
    K_int : numpy.ndarray
        Internal stress kernel [Pa/m]
    K_ext : numpy.ndarray
        External stress kernel [Pa/m]
    v_plate_vec : numpy.ndarray
        Plate velocity for all creeping patches [m/s]
    v_init : numpy.ndarray
        Initial velocity of the fault patches, in the dimensions of the rheology
    slip_taper : numpy.ndarray
        Compensating coseismic tapered slip on creeping patches [m]
    delta_tau_bounded : numpy.ndarray
        Bounded coseismic stress change [Pa]
    v_0 : float
        Reference velocity [m/s]
    alpha_h_upper : float
        Upper interface rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa]
    alpha_n_lower : float
        Lower plate interface nonlinear viscous rheology strength constant  [Pa^n * s/m]
    n_lower : float
        Lower plate interface power-law exponent [-]
    simple_rk4 : bool
        Decide whether to use the simple RK4 integrator or not

    Returns
    -------
    full_state : numpy.ndarray
        Full state variable at the end of the integration.
    """
    # initialize parameters
    n_vars_upper, n_vars_lower = 2, 2
    n_state_upper = n_vars_upper * n_creeping_upper
    n_state_lower = n_vars_lower * n_creeping_lower
    A_lower = alpha_n_lower ** (1 / n_lower)
    n_eval = t_eval.size
    n_slips = delta_tau_bounded.shape[1]

    # initialize arrays
    s_minus_upper = np.zeros((n_vars_upper - 1) * n_creeping_upper)
    s_minus_lower = np.zeros(n_creeping_lower)
    assert np.all(v_init[:n_creeping_upper] > 0)
    v_minus_upper = np.log(v_init[:n_creeping_upper] / v_0)
    # if isinstance(self.fault.upper_rheo, rheologies.RateStateSteadyLogarithmic):
    #     v_minus_upper = self.fault.upper_rheo.v2zeta(v_minus_upper)
    v_minus_lower = v_init[n_creeping_upper:]
    full_state = np.empty((n_state_upper + n_state_lower, n_eval))
    full_state[:] = np.NaN
    state_plus = np.concatenate((s_minus_upper, v_minus_upper, s_minus_lower, v_minus_lower))

    # make flat ODE function arguments
    args = (n_creeping_upper, v_plate_vec, K_int, K_ext,
            v_0, alpha_h_upper, A_lower, n_lower)

    # integrate
    spun_up = 0
    i_slip = 0
    steps = np.sort(np.concatenate((i_eq, i_break)))
    i = 0
    while i < steps.size - 1:
        # get indices
        ji, jf = steps[i], steps[i+1]
        ti, tf = t_eval[ji], t_eval[jf]
        # call integrator
        if simple_rk4:
            sol = myrk4(ti, tf, state_plus, t_eval[ji:jf + 1], *args).T
        else:
            with objmode(sol="float64[:, :]", success="boolean"):
                sol = solve_ivp(flat_ode_rdlog_plvis,
                                t_span=[ti, tf],
                                y0=state_plus,
                                t_eval=t_eval[ji:jf + 1],
                                method="RK45", rtol=1e-9, atol=1e-12, args=args)
                success = sol.success
                sol = sol.y
            if not success:
                raise RuntimeError("Integrator failed.")
        # save state to output array
        full_state[:, ji:jf + 1] = sol
        # can already stop here if this is the last interval
        if i == steps.size - 2:
            break
        # at the end of a full cycle, check the early stopping criteria
        if (not spun_up) and (i > n_slips) and (jf in i_break):
            old_full_state = full_state[:, steps[i-2*n_slips-1]:steps[i-n_slips]]
            new_full_state = full_state[:, steps[i-n_slips]:steps[i+1]]
            old_state_upper = old_full_state[:n_state_upper, :]
            old_state_lower = old_full_state[n_state_upper:, :]
            new_state_upper = new_full_state[:n_state_upper, :]
            new_state_lower = new_full_state[n_state_upper:, :]
            old_v_upper = v_0 * np.exp(old_state_upper[-n_creeping_upper:, -1])
            old_v_lower = old_state_lower[-n_creeping_lower:, -1]
            new_v_upper = v_0 * np.exp(new_state_upper[-n_creeping_upper:, -1])
            new_v_lower = new_state_lower[-n_creeping_lower:, -1]
            # if isinstance(self.fault.upper_rheo, rheologies.RateStateSteadyLogarithmic):
            #     old_v_upper = self.fault.upper_rheo.zeta2v(old_v_upper)
            #     new_v_upper = self.fault.upper_rheo.zeta2v(new_v_upper)
            lhs_upper = np.abs(old_v_upper - new_v_upper)
            lhs_lower = np.abs(old_v_lower - new_v_lower)
            rhs_upper = (1e-4) * np.abs(v_plate_vec[0]) + (1e-4) * np.abs(new_v_upper)
            rhs_lower = (1e-4) * np.abs(v_plate_vec[-1]) + (1e-4) * np.abs(new_v_lower)
            stop_now = np.all(lhs_upper <= rhs_upper) & np.all(lhs_lower <= rhs_lower)
            if stop_now:
                spun_up = jf
                # advance i to the last cycle (don't forget the general advance later)
                i = steps.size - n_slips - 3
        elif spun_up and (jf in i_break):
            break
        # apply step change only if there is one
        if (jf in i_eq):
            state_upper, state_lower = sol[:n_state_upper, -1], sol[n_state_upper:, -1]
            s_minus_upper = state_upper[:-n_creeping_upper]
            zeta_minus_upper = state_upper[-n_creeping_upper:]
            s_minus_lower = state_lower[:-n_creeping_lower]
            v_minus_lower = state_lower[-n_creeping_lower:]
            s_plus_upper = s_minus_upper.ravel().copy()
            s_plus_upper[:n_creeping_upper] += slip_taper[:, i_slip]
            s_plus_lower = s_minus_lower.ravel()
            zeta_plus_upper = get_new_vel_rdlog(zeta_minus_upper,
                                                delta_tau_bounded[:n_creeping_upper, i_slip],
                                                np.ones(n_creeping_upper) * alpha_h_upper)
            v_plus_lower = get_new_vel_plvis(v_minus_lower,
                                             delta_tau_bounded[n_creeping_upper:, i_slip],
                                             np.ones(n_creeping_upper) * alpha_n_lower,
                                             np.ones(n_creeping_upper) * n_lower,
                                             np.ones(n_creeping_upper) * A_lower)
            state_plus = np.concatenate((s_plus_upper, zeta_plus_upper,
                                         s_plus_lower, v_plus_lower))
            i_slip = (i_slip + 1) % n_slips
        else:
            state_plus = sol[:, -1]
        # advance
        i += 1

    # warn if we never spun up
    if not spun_up:
        print(f"Simulation did not spin up after {len(i_break) - 1} cycles!")

    full_state[n_creeping_upper:n_state_upper, :] = \
        v_0 * np.exp(full_state[n_creeping_upper:n_state_upper, :])
    # if isinstance(self.fault.upper_rheo, rheologies.RateStateSteadyLogarithmic):
    #     vel_upper = self.fault.upper_rheo.zeta2v(vel_upper)

    # done
    return full_state


@njit(float64[:, :](float64[:, ::1], int64, int64, float64[:, ::1], float64[:, ::1]),
      cache=True)
# optional(float64[:, ::1]), optional(float64[:, ::1])))
def get_surface_displacements_plvis_plvis(full_state, n_creeping_upper, n_creeping_lower,
                                          G_surf, deep_creep_slip):  # , locked_slip):
    """
    Calculate the surface displacements given the output of ``run``.

    Parameters
    ----------
    full_state : numpy.ndarray
        Full state variable at the end of the integration.
    n_creeping_upper : int
        Number [-] of creeping patches in the upper fault interface
    n_creeping_lower : int
        Number [-] of creeping patches in the lower fault interface
    G_surf : numpy.ndarray
        Surface displacements Green's matrix [-] (dimensions must whether `locked_slip`
        and/or `deep_creep_slip` are passed to function)
    deep_creep_slip : numpy.ndarray
        Timeseries of slip [m] on the deep creep patches
    locked_slip : numpy.ndarray, optional
        Timeseries of slip [m] on the locked patches

    Returns
    -------
    surf_disp : numpy.ndarray
        Surface displacement timeseries.
    """
    # extract timeseries from solution
    slip_upper = full_state[:n_creeping_upper, :]
    slip_lower = full_state[2 * n_creeping_upper:2 * n_creeping_upper + n_creeping_lower, :]
    # add the locked and deep patches to the combined upper & lower slip history matrix
    slips_all = np.concatenate((slip_upper, slip_lower), axis=0)
    # if locked_slip is not None:s
    #     slips_all = np.concatenate((locked_slip[:, :slip_upper.shape[1]], slips_all),
    #                                axis=0)
    # if deep_creep_slip is not None:
    slips_all = np.concatenate((slips_all, deep_creep_slip), axis=0)
    # calculate all surface displacements for last full cycle
    surf_disps = G_surf @ slips_all
    return surf_disps


class Fault2D():
    """
    Base class for the subduction fault mesh.
    """

    def __init__(self, theta, D_lock, H, nu, E, v_s, halflen,
                 upper_rheo, n_upper, lower_rheo, n_lower_left,
                 n_lower_right, halflen_factor_lower,
                 D_max=None, x1_pretrench=None):
        """
        Define the fault mesh of the subduction zone fault system, based on the
        Elastic Subducting Plate Model (ESPM) of [kanda2010]_.

        Parameters
        ----------
        theta : float
            Dip angle [rad] of the plate interface (positive).
        D_lock : float
            Locking depth [m] of the upper plate interface (positive).
        H : float
            Subducting plate thickness [m].
        nu : float
            Poisson's ratio [-] of the fault zone.
        E : float
            Young's modulus [Pa] of the fault zone.
        v_s : float
            Shear wave velocity [m/s] in the fault zone.
        halflen : float
            Fault patch half-length [m], used for all locked patches.
            If ``D_max`` and ``x1_pretrench`` are not set, this length is also used for all
            creeping patches, otherwise, this is their minimum half-length.
        upper_rheo : Rheology
            Upper plate interface's rheology.
        n_upper : int
            Number [-] of patches on upper plate interface.
        lower_rheo : Rheology
            Lower plate interface's rheology. Pass ``None`` if it should not be simulated,
            but enforced to have the plate velocity.
        n_lower_left : int
            Number [-] of patches on lower plate interface (left of the bend).
        n_lower_right : int
            Number [-] of patches on lower plate interface (right of the bend).
        halflen_factor_lower : float
            Factor used to get a different minimum half-length of the patches on the lower
            plate interface.
        D_max : float, optional
            Maximum depth [m] of the upper plate interface (positive).
            If set, this makes the mesh use linearly-increasing patch sizes away from the
            locked zone. (``x1_pretrench`` must be set as well.)
        x1_pretrench : float, optional
            Horizontal distance [m] of the lower plate interface before the trench (positive).
            If set, this makes the mesh use linearly-increasing patch sizes away from the
            locked zone. (``D_max`` must be set as well.)

        References
        ----------

        .. [kanda2010] Kanda, R. V. S., & Simons, M. (2010).
           *An elastic plate model for interseismic deformation in subduction zones.*
           Journal of Geophysical Research: Solid Earth, 115(B3).
           doi:`10.1029/2009JB006611 <https://doi.org/10.1029/2009JB006611>`_.
        """
        # initialize
        self.theta = float(theta)
        """ Subducting plate dip angle [rad] """
        assert 0 < self.theta < np.pi / 2
        self.D_lock = float(D_lock)
        """ Theoretical locking depth [m] of the upper plate interface """
        assert self.D_lock > 0
        self.H = float(H)
        """ Subducting plate thickness [m] """
        assert self.H >= 0
        self.nu = float(nu)
        """ Poisson's ratio [-] of the fault zone """
        self.E = float(E)
        """ Young's modulus [Pa] of the fault zone """
        self.halflen = float(halflen)
        """ Fault patch half-length [m] on upper interface """
        assert self.halflen > 0
        self.upper_rheo = upper_rheo
        """ Upper plate interface's rheology """
        assert isinstance(self.upper_rheo, Rheology)
        self.n_upper = int(n_upper)
        """ Number [-] of patches on upper plate interface """
        assert self.n_upper >= 1
        self.lower_rheo = lower_rheo
        """ Lower plate interface's rheology """
        assert isinstance(self.lower_rheo, Rheology) or \
            (self.lower_rheo is None)
        self.n_lower_left = int(n_lower_left)
        """ Number [-] of patches on lower plate interface (left of bend) """
        assert self.n_lower_left >= 1
        self.n_lower_right = int(n_lower_right)
        """ Number [-] of patches on lower plate interface (right of bend) """
        assert self.n_lower_right >= 1
        self.halflen_factor_lower = float(halflen_factor_lower)
        """ Prefactor [-] to change the lower interface half-length """
        assert self.halflen_factor_lower >= 1
        self.lower_halflen = self.halflen * self.halflen_factor_lower
        """ Fault patch half-length [m] on lower interface """
        if self.lower_rheo is not None:
            assert self.H >= 2 * self.lower_halflen, "Plate too thin for given patch sizes."
        self.v_s = float(v_s)
        """ Shear wave velocity [m/s] in the fault zone """
        self.mu_over_2vs = self.E / (2 * (1 + self.nu) * 2 * self.v_s)
        """ Radiation damping term [Pa * s/m] """

        # switch between constant or linearly-varying patch sizes
        if (D_max is not None) and (x1_pretrench is not None):
            D_max = float(D_max)
            x1_pretrench = float(x1_pretrench)
            assert D_max > 0
            assert x1_pretrench > 0
            variable_mesh = True
        else:
            D_max = None
            x1_pretrench = None
            variable_mesh = False
        self.D_max = D_max
        """ Maximum depth [m] of the upper plate interface (optional) """
        self.x1_pretrench = x1_pretrench
        """ Horizontal distance [m] of the lower plate interface before the trench (optional) """
        self.variable_mesh = variable_mesh
        """ Flag whether the creeping patches are linearly-varying in size, or not """

        # create mesh, centered about the x2 axis
        if self.variable_mesh:
            # project the locking depth onto dip angle
            L_lock = self.D_lock / np.sin(self.theta)
            # get number of locked and creeping patches on upper interface
            n_lock = int(L_lock // (2 * self.halflen))
            n_creep_up = self.n_upper - n_lock
            assert n_creep_up > 0, "Current geometry yields no upper creeping patches."
            # project maximum interface depth onto dip angle
            L_max = self.D_max / np.sin(self.theta)
            # get length of creeping segment that needs to be linearly varying
            delta_L = L_max - n_lock * 2 * self.halflen
            # get linear half-length increase necessary given the number of patches
            # and length of creeping segment, on all three interface regions
            delta_h_upper = ((delta_L - 2 * self.halflen * n_creep_up) /
                             (n_creep_up**2 - n_creep_up))
            delta_h_lower_right = \
                ((L_max - 2 * self.lower_halflen * self.n_lower_right) /
                 (self.n_lower_right**2 - self.n_lower_right))
            delta_h_lower_left = \
                ((self.x1_pretrench - 2 * self.lower_halflen * self.n_lower_left) /
                 (self.n_lower_left**2 - self.n_lower_left))
            # check that we're not running into numerical problems from starkly
            # increasing patch sizes
            if any([d > 0.2 for d in [delta_h_upper / self.halflen,
                                      delta_h_lower_right / self.lower_halflen,
                                      delta_h_lower_left / self.lower_halflen]]):
                raise ValueError("Half-length increase greater than 20%.")
            # build vector of half-lengths
            halflen_vec = np.concatenate([
                np.ones(n_lock) * self.halflen,
                self.halflen + np.arange(n_creep_up) * delta_h_upper,
                (self.lower_halflen + np.arange(self.n_lower_left) * delta_h_lower_left)[::-1],
                self.lower_halflen + np.arange(self.n_lower_right) * delta_h_lower_right])
        else:
            # build half-length vector from constant size
            halflen_vec = np.ones(self.n_upper + self.n_lower_left + self.n_lower_right
                                  ) * self.halflen
            halflen_vec[self.n_upper:] *= self.halflen_factor_lower
        self.halflen_vec = halflen_vec
        """ Half-lengths [m] for each patch in the fault """
        s = self.H * np.tan(self.theta / 2)
        R = np.array([[np.cos(-self.theta), -np.sin(-self.theta)],
                      [np.sin(-self.theta), np.cos(-self.theta)]])
        # upper plate interface
        upper_right_x1 = np.concatenate([[0], np.cumsum(2*self.halflen_vec[:self.n_upper])])
        upper_right_x2 = np.zeros_like(upper_right_x1)
        upper_right = R @ np.stack([upper_right_x1, upper_right_x2], axis=0)
        # lower left plate interface
        temp = self.halflen_vec[self.n_upper + self.n_lower_left - 1:self.n_upper - 1:-1]
        lower_left_x1 = -s - np.concatenate([[0], np.cumsum(2*temp)])[::-1]
        lower_left_x2 = -self.H * np.ones(self.n_lower_left + 1)
        lower_left = np.stack([lower_left_x1, lower_left_x2], axis=0)
        # lower right
        lower_right_x1 = np.concatenate([
            [0], np.cumsum(2*self.halflen_vec[self.n_upper + self.n_lower_left:])])
        lower_right_x2 = np.zeros_like(lower_right_x1)
        lower_right = (R @ np.stack([lower_right_x1, lower_right_x2], axis=0)
                       - np.array([[s], [self.H]]))
        # concatenate mesh parts
        self.end_upper = upper_right
        """ 2-element coordinates of upper fault patch endpoints [m] """
        self.end_lower = np.concatenate([lower_left, lower_right[:, 1:]], axis=1)
        """ 2-element coordinates of lower fault patch endpoints [m] """
        self.end = np.concatenate([self.end_upper, self.end_lower], axis=1)
        """ 2-element coordinates of fault patch endpoints [m] """
        self.mid = np.concatenate([upper_right[:, :-1] + upper_right[:, 1:],
                                   lower_left[:, :-1] + lower_left[:, 1:],
                                   lower_right[:, :-1] + lower_right[:, 1:]],
                                  axis=1) / 2
        """ 2-element coordinates of fault patch midpoints [m] """
        self.mid_x1 = self.mid[0, :]
        """ :math:`x_1` coordinates of fault patch midpoints [m] """
        self.mid_x2 = self.mid[1, :]
        """ :math:`x_2` coordinates of fault patch midpoints [m] """
        # access subparts
        self.ix_upper = np.arange(self.mid_x1.size) < upper_right_x1.size
        """ Mask of upper fault interface patches """
        self.ix_lower = ~self.ix_upper
        """ Mask of lower fault interface patches (if existing) """
        # locked is the part that slips coseismically on the upper plate interface
        self.x1_lock = self.D_lock / np.tan(self.theta)
        """ Theoretical surface location [m] of end of locked interface """
        ix_locked = self.mid_x1 <= self.x1_lock - self.halflen
        ix_locked[self.n_upper:] = False
        self.ix_locked = ix_locked
        """ Mask of fault patches that are locked interseismically """
        self.n_locked = (self.ix_locked).sum()
        """ Number [-] of locked patches """
        # assert self.n_locked == n_lock
        self.n_creeping = (~self.ix_locked).sum()
        """ Number [-] of creeping patches """
        self.n_creeping_upper = (~self.ix_locked[:self.n_upper]).sum()
        """ Number [-] of creeping patches in the upper fault interface """
        # assert self.n_creeping_upper == n_creep_up
        self.n_creeping_lower = self.n_creeping - self.n_creeping_upper
        """ Number [-] of creeping patches in the lower fault interface """
        assert self.n_creeping_lower == n_lower_left + n_lower_right
        self.mid_x1_locked = self.mid_x1[self.ix_locked]
        """ :math:`x_1` coordinates of locked fault patch midpoints [m] """
        self.mid_x2_locked = self.mid_x2[self.ix_locked]
        """ :math:`x_2` coordinates of locked fault patch midpoints [m] """
        self.mid_x1_creeping = self.mid_x1[~self.ix_locked]
        """ :math:`x_1` coordinates of creeping fault patch midpoints [m] """
        self.mid_x2_creeping = self.mid_x2[~self.ix_locked]
        """ :math:`x_2` coordinates of creeping fault patch midpoints [m] """
        # for later calculations, need theta and unit vectors in vector form
        theta_vec = np.ones_like(self.mid_x1) * self.theta
        theta_vec[self.n_upper:self.n_upper + self.n_lower_left] = np.pi
        theta_vec[self.n_upper + self.n_lower_left:] += np.pi
        self.theta_vec = theta_vec
        """ Plate dip angle [rad] for all fault patches """
        self.e_f = np.stack([np.sin(self.theta_vec), np.cos(self.theta_vec)], axis=0)
        """ Unit vectors [-] normal to fault patches"""
        self.e_s = np.stack([-np.cos(self.theta_vec), np.sin(self.theta_vec)], axis=0)
        """ Unit vectors [-] in fault patch slip direction """
        # get external (from the locked to the creeping patches) stress kernel
        K = Klinedisp(self.mid_x1_creeping, self.mid_x2_creeping,
                      self.mid_x1_locked, self.mid_x2_locked,
                      self.halflen_vec[self.ix_locked],
                      self.theta_vec[self.ix_locked], self.nu, self.E
                      )[:, :self.n_locked]
        Kx1x1 = K[:self.n_creeping, :]
        Kx2x2 = K[self.n_creeping:2*self.n_creeping, :]
        Kx1x2 = K[2*self.n_creeping:3*self.n_creeping, :]
        K = np.stack([Kx1x1.ravel(), Kx1x2.ravel(), Kx1x2.ravel(), Kx2x2.ravel()]
                     ).reshape(2, 2, self.n_creeping, self.n_locked).transpose(2, 3, 0, 1)
        self.K_ext = np.einsum("ki,ijkl,li->ij", self.e_s[:, ~self.ix_locked],
                               K, self.e_f[:, ~self.ix_locked], optimize=True)
        """ External stress kernel [Pa/m] """
        # get internal (within creeping patches) stress kernel
        K = Klinedisp(self.mid_x1_creeping, self.mid_x2_creeping,
                      self.mid_x1_creeping, self.mid_x2_creeping,
                      self.halflen_vec[~self.ix_locked],
                      self.theta_vec[~self.ix_locked], self.nu, self.E
                      )[:, :self.n_creeping]
        Kx1x1 = K[:self.n_creeping, :]
        Kx2x2 = K[self.n_creeping:2*self.n_creeping, :]
        Kx1x2 = K[2*self.n_creeping:3*self.n_creeping, :]
        K = np.stack([Kx1x1.ravel(), Kx1x2.ravel(), Kx1x2.ravel(), Kx2x2.ravel()]
                     ).reshape(2, 2, self.n_creeping, self.n_creeping).transpose(2, 3, 0, 1)
        self.K_int = np.einsum("ki,ijkl,li->ij", self.e_s[:, ~self.ix_locked],
                               K, self.e_f[:, ~self.ix_locked], optimize=True)
        """ Internal stress kernel [Pa/m] """
        self.n_state_upper = self.upper_rheo.n_vars * self.n_creeping_upper
        """ Size [-] of upper plate interface state variable """
        self.n_state_lower = (self.lower_rheo.n_vars * self.n_creeping_lower
                              if self.lower_rheo is not None
                              else 2 * self.n_creeping_lower)
        """ Size [-] of lower plate interface state variable """
        if (self.n_creeping_upper == 0) or (self.n_creeping_lower == 0):
            raise ValueError("Defined geometry results in zero creeping patches in "
                             "either the upper or lower plate interface.")
        # # if upper rheology is Burgers, tell it our specific shear modulus
        # if isinstance(self.upper_rheo, rheologies.LinearBurgers):
        #     self.upper_rheo.set_G(self.K_int[:self.n_creeping_upper, :self.n_creeping_upper])
        # discretized locking depth
        self.D_lock_disc = -self.end_upper[1, self.n_locked]
        """ Discretized locking depth [m] of the upper plate interface """
        self.x1_lock_disc = self.D_lock_disc / np.tan(self.theta)
        """ Discretized surface location [m] of end of locked interface """


class SubductionSimulation():
    """
    Subduction simulation container class.
    """

    def __init__(self, v_plate, n_cycles_max, n_samples_per_eq, delta_tau_max, v_max,
                 fault, Ds_0, Ds_0_logsigma, T_rec, T_rec_logsigma, D_asp_min,
                 D_asp_max, T_anchor, T_last, enforce_v_plate, largehalflen,
                 t_obs, pts_surf):
        """
        Create a subduction simulation.

        Parameters
        ----------
        v_plate : float
            Nominal far-field plate velocity, in the dimensions of the rheology
        n_cycles_max : int
            Maximum number of cycles to simulate [-]
        n_samples_per_eq : int
            Number of internal evaluation timesteps between earthquakes [-]
        delta_tau_max : float
            Maximum shear stress change [Pa] from coseismic slip on locked patches
        v_max : float
            Maximum slip velocity [m/s] on creeping patches
        fault : Fault2D
            Fault object
        Ds_0 : numpy.ndarray
            Nominal coseismic left-lateral shearing [m] of the locked fault patch(es)
        Ds_0_logsigma : numpy.ndarray
            Standard deviation of the fault slip in logarithmic space
        T_rec : numpy.ndarray
            Nominal recurrence time [a] for each earthquake
        T_rec_logsigma : numpy.ndarray
            Standard deviation of the recurrence time in logarithmic space
        D_asp_min : numpy.ndarray
            Minimum depth [m] for the asperities of each earthquake
        D_asp_max : numpy.ndarray
            Maximum depth [m] for the asperities of each earthquake
        T_anchor : str
            Anchor date where observations end
        T_last : list
            Dates of the last occurence for each earthquake (list of strings)
        enforce_v_plate : bool
            Flag whether to allow v_plate to vary or not
        largehalflen : float
            Fault patch half-length of the deep crreep patches [m]
        t_obs : numpy.ndarray, pandas.DatetimeIndex
            Observation timesteps, either as decimal years relative to the cycle start,
            or as Timestamps
        pts_surf : numpy.ndarray
            Horizontal landward observation coordinates [m] relative to the trench
        """

        # save general sequence & fault parameters
        self.v_plate = float(v_plate)
        """ Nominal far-field plate velocity, in the dimensions of the rheology """
        self.n_cycles_max = int(n_cycles_max)
        """ Maximum number of cycles to simulate [-] """
        self.n_samples_per_eq = int(n_samples_per_eq)
        """ Number of internal evaluation timesteps between earthquakes [-] """
        self.delta_tau_max = float(delta_tau_max)
        """ Maximum shear stress change [Pa] from coseismic slip on locked patches """
        self.v_max = float(v_max)
        """ Maximum slip velocity [m/s] on creeping patches """

        # define fault
        assert isinstance(fault, Fault2D)
        if not (isinstance(fault.upper_rheo, NonlinearViscous) or
                isinstance(fault.upper_rheo, RateStateSteadyLogarithmic)) or \
           not (isinstance(fault.lower_rheo, NonlinearViscous) or
                (fault.lower_rheo is None)):
            raise NotImplementedError("SubductionSimulation is only implemented for "
                                      "NonlinearViscous or RateStateSteadyLogarithmic "
                                      "rheologies in the upper interface, and NonlinearViscous "
                                      "rheology in the lower interface.")
        self.fault = fault
        """ Fault object """

        # cast earthquake slips as NumPy array
        self.Ds_0 = np.atleast_1d(Ds_0)
        """ Nominal coseismic left-lateral shearing [m] of the locked fault patch(es) """
        self.Ds_0_logsigma = np.atleast_1d(Ds_0_logsigma)
        """ Standard deviation of the fault slip in logarithmic space """
        # load recurrence times
        self.T_rec = np.atleast_1d(T_rec)
        """ Nominal recurrence time [a] for each earthquake """
        self.T_rec_logsigma = np.atleast_1d(T_rec_logsigma)
        """ Standard deviation of the recurrence time in logarithmic space """
        # load the minimum and maximum depths of the earthquakes
        self.D_asp_min = np.atleast_1d(D_asp_min)
        """ Minimum depth [m] for the asperities of each earthquake """
        self.D_asp_max = np.atleast_1d(D_asp_max)
        """ Maximum depth [m] for the asperities of each earthquake """
        assert all([D <= self.fault.D_lock for D in self.D_asp_max]), \
            f"Asperity depths {self.D_asp_max/1e3} km are deeper than the " \
            f"locking depth {self.fault.D_lock/1e3}."
        self.T_anchor = str(T_anchor)
        """ Anchor date where observations end """
        assert isinstance(T_last, list) and all([isinstance(tl, str) for tl in T_last])
        self.T_last = T_last
        """ Dates of the last occurence for each earthquake """
        # create a NumPy array that for each locked asperity has the slip per earthquake
        self.slip_mask = np.logical_and(self.fault.mid_x2_locked.reshape(-1, 1)
                                        < -self.D_asp_min.reshape(1, -1),
                                        self.fault.mid_x2_locked.reshape(-1, 1)
                                        > -self.D_asp_max.reshape(1, -1))
        """ Mask that matches each earthquake to a fault patch """
        self.T_fullcycle = np.lcm.reduce(self.T_rec)
        """ Nominal recurrence time [a] for an entire joint earthquake cycle """
        self.n_eq = self.Ds_0.size
        """ Number of distinct earthquakes in sequence """
        self.n_eq_per_asp = (self.T_fullcycle / self.T_rec).astype(int)
        """ Number of earthquakes per asperity and full cycle """

        # create realization of the slip amount and earthquake timings
        rng = np.random.default_rng()
        # first, create realizations of occurence times
        # note that this will result in a varying plate velocity rate
        # (ignore zero-slip earthquakes)
        self.T_rec_per_asp = [rng.lognormal(np.log(t), s, n) for t, s, n in
                              zip(self.T_rec, self.T_rec_logsigma, self.n_eq_per_asp)]
        """ Recurrence time [a] realization """
        self.Ds_0_per_asp = [rng.lognormal(np.log(d), s, n) if d > 0
                             else np.array([d] * n) for d, s, n in
                             zip(self.Ds_0, self.Ds_0_logsigma, self.n_eq_per_asp)]
        """ Fault slip [m] realization """

        # sanity check that in each asperity, the nominal plate rate is recovered
        self.slip_asperities = self.slip_mask.astype(int) * self.Ds_0.reshape(1, -1)
        """ Slip [m] for each earthquake in each asperity """
        v_eff_in_asp = (self.slip_asperities / self.T_rec.reshape(1, -1)).sum(axis=1)
        assert np.allclose(v_eff_in_asp, self.v_plate * 86400 * 365.25), \
            "The nominal plate rate is not recovered in all asperities.\n" \
            f"Plate velocity = {self.v_plate * 86400 * 365.25}\n" \
            f"Effective velocity in each asperity:\n{v_eff_in_asp}"

        # second, we need to shift the random realization for each earthquake
        # individually such that they all yield the same v_plate (enforced or not)
        # get the effective recurrence time as implied by the T_rec realizations
        T_fullcycle_per_asp_eff = np.array([sum(t) for t in self.T_rec_per_asp])
        # same for the effective cumulative slip
        Ds_0_fullcycle_per_asp_eff = np.array([sum(d) for d in self.Ds_0_per_asp])
        # we need to scale each individual sequence such that it implies the same
        # recurrence time and cumulative slip in each asperity
        # (again ignoring zero-slip earthquakes)
        T_fullcycle_eff_mean = np.mean(T_fullcycle_per_asp_eff)
        Ds_0_fullcycle_mean = np.ma.masked_equal(Ds_0_fullcycle_per_asp_eff, 0).mean()
        T_rec_per_asp_adj = [np.array(self.T_rec_per_asp[i]) * T_fullcycle_eff_mean
                             / T_fullcycle_per_asp_eff[i] for i in range(self.n_eq)]
        Ds_0_per_asp_adj = [np.array(self.Ds_0_per_asp[i]) * Ds_0_fullcycle_mean
                            / Ds_0_fullcycle_per_asp_eff[i] if self.Ds_0[i] > 0
                            else np.array(self.Ds_0_per_asp[i]) for i in range(self.n_eq)]
        # now each asperity has the same effective plate velocity, which can be different
        # from the nominal one - if we want to enforce the nominal plate velocity,
        # we can rescale the recurrence times again
        self.enforce_v_plate = bool(enforce_v_plate)
        """ Flag whether to allow v_plate to vary or not """
        ix_nonzero_slip = np.argmax(self.Ds_0 > 0)
        v_plate_eff = (sum(Ds_0_per_asp_adj[ix_nonzero_slip])
                       / sum(T_rec_per_asp_adj[ix_nonzero_slip]) / 86400 / 365.25)
        if self.enforce_v_plate:
            v_plate_factor = self.v_plate / v_plate_eff
            for i in range(self.n_eq):
                T_rec_per_asp_adj[i] /= v_plate_factor
            v_plate_eff = self.v_plate
        self.v_plate_eff = v_plate_eff
        """ Effective far-field plate velocity [m/s] """
        self.T_eff = sum(T_rec_per_asp_adj[0])
        """ Effective length [a] of entire earthquake sequence """

        # third, we need to create a list of earthquake dates and associated slips
        temp_slips = np.vstack([self.slip_mask[:, i].reshape(1, -1)
                                * Ds_0_per_asp_adj[i].reshape(-1, 1)
                                for i in range(self.n_eq)])
        year_offsets = [(pd.Period(self.T_anchor, "D") - pd.Period(self.T_last[i], "D")
                         ).n / 365.25 for i in range(self.n_eq)]
        eq_df_index = np.concatenate(
            [self.T_eff -
             (np.cumsum(T_rec_per_asp_adj[i]) - T_rec_per_asp_adj[i] + year_offsets[i])
             for i in range(self.n_eq)])
        # round the dates to the closest day and combine earthquakes
        eq_df_index_rounded = np.around(eq_df_index * 365.25) / 365.25
        # build a DataFrame with exact and rounded times
        eq_df = pd.DataFrame(data=temp_slips)
        eq_df["time"] = eq_df_index
        eq_df["rounded"] = eq_df_index_rounded
        # now aggregate by rounded time, keeping the minimum exact time, and summing slip
        agg_dict = {"time": "min"}
        agg_dict.update({c: "sum" for c in range(self.fault.n_locked)})
        eq_df = eq_df.groupby("rounded").agg(agg_dict)
        # convert time column to index and sort
        eq_df.set_index("time", inplace=True)
        eq_df.sort_index(inplace=True)
        assert np.allclose(eq_df.sum(axis=0), eq_df.sum(axis=0)[0])
        self.eq_df = eq_df
        """
        DataFrame with the dates [decimal year from cycle start] and slips [m]
        for each asperity
        """

        # fourth, we need to create a list of dates to use internally when evaluating
        # the earthquake cycle - this is independent of the observation dates
        i_frac_cumsum = np.concatenate([[self.eq_df.index[-1] - self.T_eff],
                                        self.eq_df.index.values])
        T_frac = np.diff(i_frac_cumsum)
        t_eval = np.concatenate(
            [np.logspace(0, np.log10(1 + T_frac[i]), self.n_samples_per_eq, endpoint=False)
             - 1 + i_frac_cumsum[i] + j*self.T_eff
             for j in range(self.n_cycles_max) for i, t in enumerate(T_frac)])
        num_neg = (t_eval < 0).sum()
        t_eval = np.roll(t_eval, -num_neg)
        t_eval[-num_neg:] += self.n_cycles_max * self.T_eff
        self.t_eval = np.sort(np.concatenate(
            [t_eval, np.arange(self.n_cycles_max + 1) * self.T_eff]))
        """ Internal evaluation timesteps [decimal years since cycle start] """
        self.n_eval = self.t_eval.size
        """ Number of internal evaluation timesteps [-] """

        # fifth, for the integration, we need the indices of the timesteps that mark either
        # an earthquake or the start of a new cycle
        self.n_slips = self.eq_df.shape[0]
        """ Number of slips in a sequence [-] """
        self.ix_break = [i*(self.n_slips * self.n_samples_per_eq + 1)
                         for i in range(self.n_cycles_max + 1)]
        """ Indices of breaks between cycles """
        self.ix_eq = [self.ix_break[i] + j * self.n_samples_per_eq - num_neg + 1
                      for i in range(self.n_cycles_max) for j in range(1, 1 + self.n_slips)]
        """ Indices of earthquakes """

        # sixth and last, for the final loop, we need a joint timesteps array between internal
        # and external (observation) timestamps, such that we can debug, check early stopping,
        # and restrict the output to the requested timeseries
        if isinstance(t_obs, pd.DatetimeIndex):
            t_obs = self.T_eff + (t_obs - pd.Timestamp(self.T_anchor)
                                  ).total_seconds().values / 86400 / 365.25
        elif isinstance(t_obs, np.ndarray):
            if np.all(t_obs < 0):
                # this format is relative to T_anchor and more stable when T_eff varies
                t_obs = self.T_eff + t_obs
            assert np.all(t_obs >= 0) and np.all(t_obs < self.T_eff), \
                f"Range of 't_obs' ({t_obs.min()}-{t_obs.max():} years) outside of " \
                f"the earthquake cycle period ({self.T_eff:} years)."
        else:
            raise ValueError("Unknown 't_obs' data type.")
        self.t_obs = t_obs
        """ Observation timesteps [decimal years since cycle start] """
        # combine all possible timesteps
        t_obs_shifted = self.t_obs + (self.n_cycles_max - 1) * self.T_eff
        self.t_eval_joint = np.unique(np.concatenate((self.t_eval, t_obs_shifted)))
        """
        Joint internal evaluation and external observation timesteps
        [decimal years since cycle start]
        """
        # get indices of each individual subset in the new timesteps array
        self.ix_break_joint = \
            np.flatnonzero(np.isin(self.t_eval_joint, self.t_eval[self.ix_break]))
        """ Indices of breaks between cycles in joint timesteps """
        self.ix_eq_joint = \
            np.flatnonzero(np.isin(self.t_eval_joint, self.t_eval[self.ix_eq]))
        """ Indices of earthquakes in joint timesteps """
        self.ix_obs_joint = \
            np.flatnonzero(np.isin(self.t_eval_joint, t_obs_shifted))
        """ Indices of observation timestamps in joint timesteps """

        # get vectors of upper plate rheology parameters
        if isinstance(self.fault.upper_rheo, RateStateSteadyLogarithmic):
            # alpha_h
            self.alpha_h_vec = \
                self.fault.upper_rheo.get_param_vectors(
                    -self.fault.mid_x2_creeping[:self.fault.n_creeping_upper])
            r""" Depth-variable :math:`(a - b) * \sigma_E` [Pa] of upper plate interface """
        elif isinstance(self.fault.upper_rheo, NonlinearViscous):
            # A, alpha_n, and n
            alpha_n_vec, n_vec, A_vec = \
                self.fault.upper_rheo.get_param_vectors(
                    -self.fault.mid_x2_creeping[:self.fault.n_creeping_upper], self.v_plate)
            self.alpha_n_vec = alpha_n_vec
            r""" Depth-variable :math:`\alpha_n` [Pa^n * s/m] of upper plate interface """
            self.n_vec = n_vec
            r""" Depth-variable :math:`n` [-] of upper plate interface """
            self.A_vec = A_vec
            r""" Depth-variable :math:`A ` [Pa * (s/m)^(1/n)] of upper plate interface """
        else:
            raise NotImplementedError

        # get unbounded delta_tau
        self.delta_tau_unbounded = self.fault.K_ext @ self.eq_df.values.T
        """ Unbounded coseismic stress change [Pa] """
        # get pseudoinverse of K_int for tapered slip
        self.K_int_inv_upper = np.linalg.pinv(
            self.fault.K_int[:self.fault.n_creeping_upper, :self.fault.n_creeping_upper])
        """ Inverse of K_int [m/Pa] """
        self.delta_tau_max_from_v_max_lower = \
            ((self.fault.lower_rheo.alpha_n * self.v_max)**(1 / self.fault.lower_rheo.n) -
             (self.fault.lower_rheo.alpha_n * self.v_plate)**(1 / self.fault.lower_rheo.n)
             if self.fault.lower_rheo is not None else np.inf)
        """ Maximum shear stress change [Pa] in lower plate from capped velocity """
        if isinstance(self.fault.upper_rheo, NonlinearViscous):
            delta_tau_max_from_v_max_upper = \
                (self.alpha_n_vec * self.v_max)**(1 / self.n_vec) - \
                (self.alpha_n_vec * self.v_plate)**(1 / self.n_vec)
        elif isinstance(self.fault.upper_rheo, RateStateSteadyLogarithmic):
            delta_tau_max_from_v_max_upper = self.alpha_h_vec * \
                (np.log(self.v_max / self.fault.upper_rheo.v_0) -
                 np.log(self.v_plate / self.fault.upper_rheo.v_0))
        self.delta_tau_max_from_v_max_upper = delta_tau_max_from_v_max_upper
        """ Maximum shear stress change [Pa] in upper plate from capped velocity """
        self.delta_tau_max_joint_upper = np.fmin(self.delta_tau_max,
                                                 self.delta_tau_max_from_v_max_upper)
        """ Joint maximum shear stress change [Pa] allowed in upper plate """
        self.delta_tau_max_joint_lower = \
            (min(self.delta_tau_max, self.delta_tau_max_from_v_max_lower)
             if self.fault.lower_rheo is not None else np.inf)
        """ Joint maximum shear stress change [Pa] allowed in lower plate """
        # create tapered slip by making delta_tau linearly increase until delta_tau_max
        delta_tau_bounded = self.delta_tau_unbounded.copy()
        delta_tau_bounded[:self.fault.n_creeping_upper, :] = \
            np.fmin(self.delta_tau_max_joint_upper.reshape(-1, 1),
                    self.delta_tau_unbounded[:self.fault.n_creeping_upper, :])
        self.delta_tau_bounded = delta_tau_bounded
        """ Bounded coseismic stress change [Pa] """
        # get the additional slip
        self.slip_taper = (self.K_int_inv_upper @
                           (self.delta_tau_bounded - self.delta_tau_unbounded
                            )[:self.fault.n_creeping_upper, :])
        # check if the lower plate should have been bounded as well
        if self.fault.lower_rheo is not None:
            assert not np.any(np.abs(self.delta_tau_bounded[self.fault.n_creeping_upper:, :])
                              > self.delta_tau_max_joint_lower), \
                ("Maximum stress change delta_tau_bounded "
                 f"{np.max(np.abs(self.delta_tau_bounded)):.2e} Pa in lower interface "
                 f"above delta_tau_max = {self.delta_tau_max_joint_lower:.2e} Pa")
        self.slip_taper_ts = \
            pd.DataFrame(index=self.eq_df.index, data=self.slip_taper.T) \
            .cumsum(axis=0).reindex(index=self.t_obs, method="ffill", fill_value=0)
        """ Timeseries of tapered slip [m] on the upper creeping fault patches """

        # need the imagined location and orientation of the deep creep patches
        self.largehalflen = float(largehalflen)
        """ Fault patch half-length of the deep crreep patches [m] """
        self.mid_deep_x1 = \
            np.array([self.fault.mid_x1[self.fault.n_upper - 1]
                      + np.cos(self.fault.theta_vec[self.fault.n_upper - 1])
                      * self.fault.halflen_vec[self.fault.n_upper - 1]
                      + np.cos(self.fault.theta_vec[self.fault.n_upper - 1])
                      * self.largehalflen,
                      self.fault.mid_x1[self.fault.n_upper + self.fault.n_lower_left - 1]
                      - self.fault.halflen_vec[self.fault.n_upper + self.fault.n_lower_left - 1]
                      - self.largehalflen,
                      self.fault.mid_x1[-1]
                      + np.cos(self.fault.theta_vec[-1] - np.pi)
                      * self.fault.halflen_vec[-1]
                      + np.cos(self.fault.theta_vec[-1] - np.pi)
                      * self.largehalflen])
        """ :math:`x_1` coordinates of deep creep fault patch midpoints [m] """
        self.mid_deep_x2 = \
            np.array([self.fault.mid_x2[self.fault.n_upper - 1]
                      - np.sin(self.fault.theta_vec[self.fault.n_upper - 1])
                      * self.fault.halflen_vec[self.fault.n_upper - 1]
                      - np.sin(self.fault.theta_vec[self.fault.n_upper - 1])
                      * self.largehalflen,
                      self.fault.mid_x2[self.fault.n_upper + self.fault.n_lower_left - 1],
                      self.fault.mid_x2[-1]
                      - np.sin(self.fault.theta_vec[-1] - np.pi)
                      * self.fault.halflen_vec[-1]
                      - np.sin(self.fault.theta_vec[-1] - np.pi)
                      * self.largehalflen])
        """ :math:`x_2` coordinates of deep creep fault patch midpoints [m] """
        self.theta_vec_deep = \
            np.array([self.fault.theta_vec[self.fault.n_upper - 1],
                      np.pi,
                      self.fault.theta_vec[-1]])
        """ Plate dip angle [rad] for deep creep fault patches """

        # create the Green's matrices
        self.pts_surf = pts_surf
        """ :math:`x_1` coordinates of surface observation points [m] """
        self.n_stations = self.pts_surf.size
        """ Number of surface observing stations """
        self.G_surf_fault = Glinedisp(
            self.pts_surf, 0, self.fault.mid_x1, self.fault.mid_x2,
            self.fault.halflen_vec, self.fault.theta_vec, self.fault.nu
            )[:, :self.fault.mid_x1.size]
        """ Green's matrix [-] relating slip on the main fault patches to surface motion """
        self.G_surf_deep = Glinedisp(
            self.pts_surf, 0, self.mid_deep_x1, self.mid_deep_x2,
            self.largehalflen, self.theta_vec_deep, self.fault.nu)[:, :3]
        """ Green's matrix [-] relating slip on the deep creep patches to surface motion """
        self.G_surf = np.hstack([self.G_surf_fault, self.G_surf_deep])
        """ Joint Green's matrix [-] relating slip on the entire ESPM to surface motion """

        # calculate the best initial velocity state from the steady state ODE
        v_plate_vec = np.ones(self.fault.n_creeping) * self.v_plate
        v_plate_vec[self.fault.n_creeping_upper:] *= -1
        self.v_plate_vec = v_plate_vec
        """ Vector with the plate velocity for each creeping patch [m/s] """
        # get the initial velocity, taking advantage of the option that there could be a
        # deep transition zone
        v_init = v_plate_vec.copy()
        if self.fault.upper_rheo.deep_transition is not None:
            ix_deep = np.argmin(np.abs(-self.fault.mid_x2_creeping[:self.fault.n_creeping_upper]
                                       - self.fault.upper_rheo.deep_transition
                                       - self.fault.upper_rheo.deep_transition_width))
            if isinstance(self.fault.upper_rheo, RateStateSteadyLogarithmic):
                v_init[:ix_deep] = np.linspace(self.v_plate * 1e-6, self.v_plate,
                                               num=ix_deep, endpoint=False)
            elif isinstance(self.fault.upper_rheo, NonlinearViscous):
                v_init[:ix_deep] = np.linspace(0, self.v_plate, num=ix_deep, endpoint=False)
        self.v_init = v_init
        """ Initial velocity in all creeping patches [m/s] """

    @property
    def locked_slip(self):
        """ Timeseries of slip [m] on the locked patches for observation timespan """
        return self.eq_df.cumsum(axis=0) \
            .reindex(index=self.t_obs, method="ffill", fill_value=0).values.T

    @property
    def deep_creep_slip(self):
        """ Timeseries of slip [m] on the deep creep patches for observation timestamps """
        return (np.tile(self.t_obs.reshape(1, -1), (3, 1))
                * np.array([1, -1, -1]).reshape(3, 1)
                * self.v_plate_eff * 86400 * 365.25)

    @staticmethod
    def read_config_file(config_file):
        """
        Read a configuration file and return it as a parsed dictionary.

        Parameters
        ----------
        config_file : str
            Path to INI configuration file.

        Returns
        -------
        cfg_dict : dict
            Parsed configuration file.
        """

        # load configuration file
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        with open(config_file, mode="rt") as f:
            cfg.read_file(f)
        cfg_seq, cfg_fault, cfg_mesh = cfg["sequence"], cfg["fault"], cfg["mesh"]

        # parse rheologies
        upper_rheo_dict = dict(cfg["upper_rheo"])
        upper_rheo_type = upper_rheo_dict.pop("type")
        upper_rheo_kw_args = {k: float(v) for k, v in upper_rheo_dict.items()}
        try:
            lower_rheo_dict = dict(cfg["lower_rheo"])
        except KeyError:
            lower_rheo_type = None
            lower_rheo_kw_args = None
        else:
            lower_rheo_type = lower_rheo_dict.pop("type")
            lower_rheo_kw_args = {k: float(v) for k, v in lower_rheo_dict.items()}

        # parse everything else
        cfg_dict = {
            "theta": np.deg2rad(cfg_fault.getfloat("theta_deg")),
            "D_lock": cfg_fault.getfloat("D_lock"),
            "H": cfg_fault.getfloat("H"),
            "nu": cfg_fault.getfloat("nu"),
            "E": cfg_fault.getfloat("E"),
            "v_s": cfg_fault.getfloat("v_s"),
            "halflen": cfg_mesh.getfloat("halflen"),
            "n_upper": cfg_mesh.getint("n_up"),
            "n_lower_left": cfg_mesh.getint("n_low_l"),
            "n_lower_right": cfg_mesh.getint("n_low_r"),
            "halflen_factor_lower": cfg_mesh.getfloat("halflen_factor_lower"),
            "D_max": cfg_mesh.getfloat("D_max", fallback=None),
            "x1_pretrench": cfg_mesh.getfloat("x1_pretrench", fallback=None),
            "v_plate": cfg_seq.getfloat("v_plate"),
            "n_cycles_max": cfg_seq.getint("n_cycles_max"),
            "n_samples_per_eq": cfg_seq.getint("n_samples_per_eq"),
            "delta_tau_max": cfg_fault.getfloat("delta_tau_max", fallback=np.inf),
            "v_max": cfg_fault.getfloat("v_max", fallback=np.inf),
            "Ds_0": np.atleast_1d(json.loads(cfg_seq["Ds_0"])),
            "Ds_0_logsigma": np.atleast_1d(json.loads(cfg_seq["Ds_0_logsigma"])),
            "T_rec": np.atleast_1d(json.loads(cfg_seq["T_rec"])),
            "T_rec_logsigma": np.atleast_1d(json.loads(cfg_seq["T_rec_logsigma"])),
            "D_asp_min": np.atleast_1d(json.loads(cfg_seq["D_asp_min"])),
            "D_asp_max": np.atleast_1d(json.loads(cfg_seq["D_asp_max"])),
            "T_anchor": cfg_seq.get("T_anchor"),
            "T_last": json.loads(cfg_seq["T_last"]),
            "enforce_v_plate": cfg_seq.getboolean("enforce_v_plate"),
            "largehalflen": cfg_mesh.getfloat("largehalflen"),
            "upper_rheo_type": upper_rheo_type,
            "lower_rheo_type": lower_rheo_type,
            "upper_rheo_kw_args": upper_rheo_kw_args,
            "lower_rheo_kw_args": lower_rheo_kw_args
            }
        return cfg_dict

    @classmethod
    def from_config_dict(cls, cfg, t_obs, pts_surf):
        """
        Create a SubductionSimulation object from a configuration dictionary.

        Parameters
        ----------
        cfg : dict
            Dictionary containing all parsed elements from the configuration file
        t_obs : numpy.ndarray, pandas.DatetimeIndex
            Observation timesteps, either as decimal years relative to the cycle start,
            or as Timestamps
        pts_surf : numpy.ndarray
            Horizontal landward observation coordinates [m] relative to the trench

        See Also
        --------
        read_config_file : To load a configuration file into a dictionary.
        """

        # create rheology objects
        upper_rheo = globals()[cfg["upper_rheo_type"]](**cfg["upper_rheo_kw_args"])
        if cfg["lower_rheo_type"] is None:
            lower_rheo = None
        else:
            lower_rheo = globals()[cfg["lower_rheo_type"]](**cfg["lower_rheo_kw_args"])

        # create fault object
        fault = Fault2D(theta=cfg["theta"],
                        D_lock=cfg["D_lock"],
                        H=cfg["H"],
                        nu=cfg["nu"],
                        E=cfg["E"],
                        v_s=cfg["v_s"],
                        halflen=cfg["halflen"],
                        upper_rheo=upper_rheo,
                        n_upper=cfg["n_upper"],
                        lower_rheo=lower_rheo,
                        n_lower_left=cfg["n_lower_left"],
                        n_lower_right=cfg["n_lower_right"],
                        halflen_factor_lower=cfg["halflen_factor_lower"],
                        D_max=cfg["D_max"],
                        x1_pretrench=cfg["x1_pretrench"])

        # create simulation object
        return cls(v_plate=cfg["v_plate"],
                   n_cycles_max=cfg["n_cycles_max"],
                   n_samples_per_eq=cfg["n_samples_per_eq"],
                   delta_tau_max=cfg["delta_tau_max"],
                   v_max=cfg["v_max"],
                   fault=fault,
                   Ds_0=cfg["Ds_0"],
                   Ds_0_logsigma=cfg["Ds_0_logsigma"],
                   T_rec=cfg["T_rec"],
                   T_rec_logsigma=cfg["T_rec_logsigma"],
                   D_asp_min=cfg["D_asp_min"],
                   D_asp_max=cfg["D_asp_max"],
                   T_anchor=cfg["T_anchor"],
                   T_last=cfg["T_last"],
                   enforce_v_plate=cfg["enforce_v_plate"],
                   largehalflen=cfg["largehalflen"],
                   t_obs=t_obs,
                   pts_surf=pts_surf)

    @staticmethod
    def get_n(alpha_n, alpha_eff, v_eff):
        r"""
        Calculate the real linear viscous strength constant from the effective one.

        Parameters
        ----------
        alpha_n : float
            Nonlinear viscous rheology strength constant :math:`\alpha_n` [Pa^n * s/m]
        alpha_eff : float
            Effective linear viscous strength constant [Pa * s/m]
        v_eff : float
            Effective velocity [m/s] used for ``alpha_eff`` conversions

        Returns
        -------
        n : float
            Power-law exponent :math:`n` [-]
        """
        return (np.log(alpha_n) + np.log(v_eff)) / (np.log(alpha_eff) + np.log(v_eff))

    @staticmethod
    def get_alpha_n(alpha_eff, n, v_eff):
        r"""
        Calculate the real linear viscous strength constant from the effective one.

        Parameters
        ----------
        alpha_eff : float
            Effective linear viscous strength constant [Pa * s/m]
        n : float
            Power-law exponent :math:`n` [-]
        v_eff : float
            Effective velocity [m/s] used for ``alpha_eff`` conversions

        Returns
        -------
        alpha_n : float
            Nonlinear viscous rheology strength constant :math:`\alpha_n` [Pa^n * s/m]
        """
        alpha_n = alpha_eff**n * v_eff**(n-1)
        return alpha_n

    @staticmethod
    def get_alpha_eff(alpha_n, n, v_eff):
        r"""
        Calculate the effective linear viscous strength constant from the real one.

        Parameters
        ----------
        alpha_n : float
            Nonlinear viscous rheology strength constant :math:`\alpha_n` [Pa^n * s/m]
        n : float
            Power-law exponent :math:`n` [-]
        v_eff : float
            Effective velocity [m/s] used for ``alpha_eff`` conversions

        Returns
        -------
        alpha_eff : float
            Effective linear viscous strength constant [Pa * s/m]
        """
        if isinstance(v_eff, np.ndarray):
            temp = v_eff.copy()
            temp[temp == 0] = np.NaN
        else:
            temp = v_eff
        alpha_eff = alpha_n**(1/n) * temp**((1-n)/n)
        return alpha_eff

    @staticmethod
    def get_alpha_eff_from_alpha_h(alpha_h, v_eff):
        r"""
        Calculate the effective viscosity from the rate-dependent friction.

        Parameters
        ----------
        alpha_h : float
            Rate-and-state parameter :math:`(a - b) * \sigma_E`,
            where :math:`a` and :math:`b` [-] are the rate-and-state frictional properties,
            and :math:`\sigma_E` [Pa] is effective fault normal stress.
        v_eff : float
            Effective velocity [m/s] used for ``alpha_eff`` conversions

        Returns
        -------
        alpha_eff : float
            Effective linear viscous strength constant [Pa * s/m]
        """
        if isinstance(v_eff, np.ndarray):
            temp = v_eff.copy()
            temp[temp == 0] = np.NaN
        else:
            temp = v_eff
        alpha_eff = alpha_h / temp
        return alpha_eff

    def run(self, simple_rk4=False):
        """
        Run a full simulation.
        """
        # run forward integration
        if self.fault.lower_rheo is None:
            if isinstance(self.fault.upper_rheo, RateStateSteadyLogarithmic):
                full_state = flat_run_rdlog(
                    self.t_eval_joint * 86400 * 365.25, self.ix_break_joint, self.ix_eq_joint,
                    self.fault.n_creeping_upper, self.fault.n_creeping_lower, self.fault.K_int,
                    self.fault.K_ext, self.v_plate_vec, self.v_init, self.slip_taper,
                    self.delta_tau_bounded, self.fault.upper_rheo.v_0, self.alpha_h_vec,
                    self.fault.mu_over_2vs)
            elif isinstance(self.fault.upper_rheo, NonlinearViscous):
                full_state = flat_run_plvis(
                    self.t_eval_joint * 86400 * 365.25, self.ix_break_joint, self.ix_eq_joint,
                    self.fault.n_creeping_upper, self.fault.n_creeping_lower, self.fault.K_int,
                    self.fault.K_ext, self.v_plate_vec, self.v_init, self.slip_taper,
                    self.delta_tau_bounded, self.alpha_n_vec, self.n_vec, self.A_vec,
                    self.fault.mu_over_2vs)
            else:
                raise NotImplementedError
        elif isinstance(self.fault.lower_rheo, NonlinearViscous):
            if isinstance(self.fault.upper_rheo, NonlinearViscous):
                full_state = flat_run_plvis_plvis(
                    self.t_eval_joint * 86400 * 365.25, self.ix_break_joint, self.ix_eq_joint,
                    self.fault.n_creeping_upper, self.fault.n_creeping_lower, self.fault.K_int,
                    self.fault.K_ext, self.v_plate_vec, self.v_init, self.slip_taper,
                    self.delta_tau_bounded, self.fault.upper_rheo.alpha_n,
                    self.fault.upper_rheo.n, self.fault.lower_rheo.alpha_n,
                    self.fault.lower_rheo.n, simple_rk4)
            elif isinstance(self.fault.upper_rheo, RateStateSteadyLogarithmic):
                full_state = flat_run_rdlog_plvis(
                    self.t_eval_joint * 86400 * 365.25, self.ix_break_joint, self.ix_eq_joint,
                    self.fault.n_creeping_upper, self.fault.n_creeping_lower, self.fault.K_int,
                    self.fault.K_ext, self.v_plate_vec, self.v_init, self.slip_taper,
                    self.delta_tau_bounded, self.fault.upper_rheo.v_0,
                    self.fault.upper_rheo.alpha_h, self.fault.lower_rheo.alpha_n,
                    self.fault.lower_rheo.n, simple_rk4)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # extract the observations that were actually requested
        obs_state = full_state[:, self.ix_obs_joint].copy()
        # since we're only calculating transient surface displacements, need to
        # remove the tapered slip due to bounded stresses
        obs_state[:self.fault.n_creeping_upper, :] -= self.slip_taper_ts.values.T
        # convert to surface displacements
        surf_disps = get_surface_displacements_plvis_plvis(
            obs_state, self.fault.n_creeping_upper, self.fault.n_creeping_lower,
            np.ascontiguousarray(self.G_surf[:, self.fault.n_locked:]),
            self.deep_creep_slip)
        return full_state, obs_state, surf_disps

    def zero_obs_at_eq(self, surf_disps):
        """
        Reset to zero the surface displacement timeseries every time an earthquake happens.
        """
        obs_zeroed = surf_disps.copy()
        slips_obs = np.logical_and(self.t_obs.min() <= self.eq_df.index,
                                   self.t_obs.max() > self.eq_df.index)
        n_slips_obs = slips_obs.sum()
        if n_slips_obs == 0:
            obs_zeroed -= obs_zeroed[:, 0].reshape(-1, 1)
        else:
            i_slips_obs = [np.argmax(self.t_obs >= t_eq) for t_eq
                           in self.eq_df.index.values[slips_obs]]
            obs_zeroed[:, :i_slips_obs[0]] -= obs_zeroed[:, i_slips_obs[0] - 1].reshape(-1, 1)
            obs_zeroed[:, i_slips_obs[0]:] -= obs_zeroed[:, i_slips_obs[0]].reshape(-1, 1)
            for i in range(1, n_slips_obs):
                obs_zeroed[:, i_slips_obs[i]:] -= obs_zeroed[:, i_slips_obs[i]].reshape(-1, 1)
        return obs_zeroed

    def _reduce_full_state(self, data):
        # get all NaN columns
        cols_all_nan = np.all(np.isnan(data), axis=0)
        # check if there was early stopping
        if cols_all_nan.sum() > 0:
            # get the border indices where integrations have been skipped
            ix_last, ix_first = np.flatnonzero(cols_all_nan)[[0, -1]]
            ix_last -= 1
            ix_first += 1
            # get indices before and after the NaN period
            ix_valid = np.r_[0:ix_last, ix_first:self.t_eval_joint.size]
            # subset data
            data = data[:, ix_valid]
            t_sub = self.t_eval_joint[ix_valid].copy()
            t_sub[ix_last:] -= self.t_eval_joint[ix_first] - self.t_eval_joint[ix_last]
            n_cyc_completed = int(np.round(self.t_eval_joint[ix_last] / self.T_eff)) + 1
        else:
            t_sub = self.t_eval_joint.copy()
            n_cyc_completed = self.n_cycles_max + 1
        # done
        return data, t_sub, n_cyc_completed

    def plot_surface_displacements(self, obs_zeroed, obs_noisy=None):
        """
        Plot the observers' surface displacement timeseries.

        Parameters
        ----------
        obs_zeroed : numpy.ndarray
            Surface displacements as output by :meth:`~zero_obs_at_eq`.
        obs_noisy : numpy.ndarray, optional
            Noisy surface observations.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        # some helper variables
        isort = np.argsort(self.pts_surf)
        i_off = 3 * np.std(obs_zeroed.ravel())
        # get float dates of observed earthquakes
        slips_obs = np.logical_and(self.t_obs.min() <= self.eq_df.index,
                                   self.t_obs.max() > self.eq_df.index)
        n_slips_obs = slips_obs.sum()
        if n_slips_obs > 0:
            i_slips_obs = [np.argmax(self.t_obs >= t_eq) for t_eq
                           in self.eq_df.index.values[slips_obs]]
            t_last_slips = [self.t_obs[islip] for islip in i_slips_obs]
        else:
            t_last_slips = []
        # start plot
        fig, ax = plt.subplots(nrows=2, sharex=True, layout="constrained")
        for tslip in t_last_slips:
            ax[0].axvline(tslip, c="0.7", zorder=-1)
            ax[1].axvline(tslip, c="0.7", zorder=-1)
        for i, ix in enumerate(isort):
            if obs_noisy is not None:
                ax[0].plot(self.t_obs, obs_noisy[ix, :] + i*i_off,
                           ".", c="k", rasterized=True)
                ax[1].plot(self.t_obs, obs_noisy[ix + self.n_stations, :] + i*i_off,
                           ".", c="k", rasterized=True)
            ax[0].plot(self.t_obs, obs_zeroed[ix, :] + i*i_off, c=f"C{i}")
            ax[1].plot(self.t_obs, obs_zeroed[ix + self.n_stations, :] + i*i_off, c=f"C{i}")
        ax[1].set_xlabel("Time")
        ax[0].set_ylabel("Horizontal [m]")
        ax[1].set_ylabel("Vertical [m]")
        fig.suptitle("Surface Displacement")
        return fig, ax

    def plot_fault_velocities(self, full_state):
        """
        Plot the velocities on all creeping fault patches.

        Parameters
        ----------
        full_state : numpy.ndarray
            State matrix as output from :meth:`~run`.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
        from cmcrameri import cm
        # extract velocities
        vels = full_state[np.r_[self.fault.n_creeping_upper:self.fault.n_state_upper,
                                self.fault.n_state_upper + self.fault.n_creeping_lower:
                                self.fault.n_state_upper + self.fault.n_state_lower],
                          :] / self.v_plate
        # check whether the simulation spun up, and NaN data needs to be skipped
        vels, t_sub, n_cyc_completed = self._reduce_full_state(vels)
        # normalize time
        t_sub /= self.T_eff
        # prepare plot
        norm = SymLogNorm(linthresh=1, vmin=-1, vmax=100)
        if self.fault.lower_rheo is None:
            fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
            ax = [ax]
        else:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 5), layout="constrained")
        # plot velocities
        c = ax[0].pcolormesh(t_sub,
                             self.fault.end_upper[0, self.fault.n_locked:] / 1e3,
                             vels[:self.fault.n_creeping_upper, :-1],
                             norm=norm, cmap=cm.vik, shading="flat")
        ax[0].set_yticks(self.fault.end_upper[0, [self.fault.n_locked, -1]] / 1e3)
        # add vertical lines for cycle breaks
        for n in range(1, n_cyc_completed):
            ax[0].axvline(n, c="k", lw=1)
        # make the y-axis increasing downwards to mimic depth even though we're plotting x1
        ax[0].invert_yaxis()
        # repeat for lower interface, if simulated
        if self.fault.lower_rheo is not None:
            c = ax[1].pcolormesh(t_sub,
                                 self.fault.end_lower[0, :] / 1e3,
                                 -vels[self.fault.n_creeping_upper:, :-1],
                                 norm=norm, cmap=cm.vik, shading="flat")
            ax[1].set_yticks(self.fault.end_lower[0, [0, -1]] / 1e3)
            # add horizontal lines to show where the lower interface is below the locked zone
            ax[1].axhline(0, c="k", lw=1)
            ax[1].axhline(self.fault.x1_lock / 1e3, c="k", lw=1)
            for n in range(1, n_cyc_completed):
                ax[1].axvline(n, c="k", lw=1)
            ax[1].invert_yaxis()
        # finish figure
        if self.fault.lower_rheo is None:
            ax[0].set_ylabel("Upper Interface\n$x_1$ [km]")
            ax[0].set_xlabel("Normalized Time $t/T$")
        else:
            ax[0].set_ylabel("Upper Interface\n$x_1$ [km]")
            ax[1].set_ylabel("Lower Interface\n$x_1$ [km]")
            ax[1].set_xlabel("Normalized Time $t/T$")
        fig.colorbar(c, ax=ax, location="right", orientation="vertical", fraction=0.05,
                     label="$v/v_{plate}$")
        fig.suptitle("Normalized Fault Patch Velocities")
        return fig, ax

    def plot_fault_slip(self, full_state, deficit=True, include_locked=True, include_deep=True):
        """
        Plot the cumulative slip (deficit) for the fault patches.

        Parameters
        ----------
        full_state : numpy.ndarray
            State matrix as output from :meth:`~run`.
        deficit : bool, optional
            If ``True`` (default), remove the plate velocity to plot slip deficit,
            otherwise keep it included.
        include_locked : bool, optional
            If ``True`` (default), also plot the slip on the locked patches.
        include_deep : bool, optional
            If ``True`` (default), also plot the slip on the semi-infinite patches
            at the end of the interfaces.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, SymLogNorm
        from cmcrameri import cm
        # extract slip
        slip = full_state[np.r_[:self.fault.n_creeping_upper,
                                self.fault.n_state_upper:
                                self.fault.n_state_upper + self.fault.n_creeping_lower], :]
        # check whether the simulation spun up, and NaN data needs to be skipped
        slip, t_sub, n_cyc_completed = self._reduce_full_state(slip)
        # normalize to slip per full cycle
        cum_slip_per_cycle = self.v_plate_eff * self.T_eff * 86400 * 365.25
        slip /= cum_slip_per_cycle
        # add optional slip histories, if desired
        if include_locked:
            eq_df_joint = pd.DataFrame(
                index=(self.eq_df.index.values.reshape(1, -1)
                       + self.T_eff * np.arange(n_cyc_completed).reshape(-1, 1)
                       ).ravel(),
                data=np.tile(self.eq_df.values, (n_cyc_completed, 1)))
            locked_slip = eq_df_joint.cumsum(axis=0) \
                .reindex(index=t_sub, method="ffill", fill_value=0).values.T
            locked_slip /= cum_slip_per_cycle
        if include_deep:
            deep_creep_slip = (np.tile(t_sub.reshape(1, -1), (3, 1))
                               * np.array([1, -1, -1]).reshape(3, 1)
                               * self.v_plate_eff * 86400 * 365.25)
            deep_creep_slip /= cum_slip_per_cycle
        # remove plate velocity to get slip deficit, if desired
        if deficit:
            cmap = cm.vik
            norm = SymLogNorm(linthresh=1e-2, vmin=-1, vmax=1)
            slip[:self.fault.n_creeping_upper] -= t_sub.reshape(1, -1) / self.T_eff
            slip[self.fault.n_creeping_upper:] += t_sub.reshape(1, -1) / self.T_eff
            slip -= slip[:, -2].reshape(-1, 1)
            if include_locked:
                locked_slip -= t_sub.reshape(1, -1) / self.T_eff
            if include_deep:
                deep_creep_slip -= (t_sub.reshape(1, -1)
                                    * np.array([1, -1, -1]).reshape(3, 1)) / self.T_eff
        else:
            norm = Normalize(vmin=0, vmax=n_cyc_completed)
            cmap = cm.batlow
        # normalize time
        t_sub /= self.T_eff
        # prepare figure
        nrows = (1 + int(self.fault.lower_rheo is not None)
                 + int(include_locked) + int(include_deep) * 3)
        hr_locked = ((self.fault.end_upper[0, self.fault.n_locked] - self.fault.end_upper[0, 0])
                     / (self.fault.end_lower[0, -1] - self.fault.end_lower[0, 0]))
        hr_lower = ((self.fault.end_lower[0, -1] - self.fault.end_lower[0, 0])
                    / (self.fault.end_upper[0, -1] - self.fault.end_upper[0, self.fault.n_locked]))
        hr = ([hr_locked] * int(include_locked) + [1]
              + [hr_locked, hr_locked] * int(include_deep)
              + [hr_lower] * int(self.fault.lower_rheo is not None)
              + [hr_locked] * int(include_deep))
        fig, ax = plt.subplots(nrows=nrows, sharex=True, gridspec_kw={"height_ratios": hr},
                               figsize=(10, 5), layout="constrained")
        iax = 0
        # plot locked
        if include_locked:
            c = ax[iax].pcolormesh(t_sub,
                                   self.fault.end_upper[0, :self.fault.n_locked + 1] / 1e3,
                                   locked_slip[:, :-1],
                                   norm=norm, cmap=cmap, shading="flat")
            ax[iax].set_ylabel("Locked\n$x_1$ [km]")
            temp_x1 = self.fault.end_upper[0, [0, self.fault.n_locked]] / 1e3
            ax[iax].set_yticks(temp_x1, [f"{x:.0f}" for x in temp_x1])
            iax += 1
        # plot upper creeping
        c = ax[iax].pcolormesh(t_sub,
                               self.fault.end_upper[0, self.fault.n_locked:] / 1e3,
                               slip[:self.fault.n_creeping_upper, :-1],
                               norm=norm, cmap=cmap, shading="flat")
        ax[iax].set_ylabel("Creeping\n$x_1$ [km]")
        temp_x1 = self.fault.end_upper[0, [self.fault.n_locked, -1]] / 1e3
        ax[iax].set_yticks(temp_x1, [f"{x:.0f}" for x in temp_x1])
        iax += 1
        # plot end patch on upper interface
        if include_deep:
            temp_x1 = np.array([self.fault.end_upper[0, -1],
                                self.mid_deep_x1[0]]) / 1e3
            c = ax[iax].pcolormesh(t_sub,
                                   temp_x1,
                                   deep_creep_slip[0, :-1].reshape(1, -1),
                                   norm=norm, cmap=cmap, shading="flat")
            ax[iax].set_ylabel("Deep Creep\n$x_1$ [km]")
            ax[iax].set_yticks(temp_x1, [f"{temp_x1[0]:.0f}", "$-\\infty$"])
            iax += 1
        # plot left end patch on lower interface
        if include_deep:
            temp_x1 = np.array([self.mid_deep_x1[1],
                                self.fault.end_lower[0, 0]]) / 1e3
            c = ax[iax].pcolormesh(t_sub,
                                   temp_x1,
                                   -deep_creep_slip[1, :-1].reshape(1, -1),
                                   norm=norm, cmap=cmap, shading="flat")
            ax[iax].set_ylabel("Deep Creep\n$x_1$ [km]")
            ax[iax].set_yticks(temp_x1, ["$-\\infty$", f"{temp_x1[1]:.0f}"])
            iax += 1
        # plot lower creeping
        if self.fault.lower_rheo is not None:
            c = ax[iax].pcolormesh(t_sub,
                                   self.fault.end_lower[0, :] / 1e3,
                                   -slip[self.fault.n_creeping_upper:, :-1],
                                   norm=norm, cmap=cmap, shading="flat")
            ax[iax].axhline(0, c="k", lw=1)
            ax[iax].axhline(self.fault.x1_lock / 1e3, c="k", lw=1)
            ax[iax].set_ylabel("Creeping\n$x_1$ [km]")
            temp_x1 = self.fault.end_lower[0, [0, -1]] / 1e3
            ax[iax].set_yticks(temp_x1, [f"{x:.0f}" for x in temp_x1])
            iax += 1
        # plot right end patch on lower interface
        if include_deep:
            temp_x1 = np.array([self.fault.end_lower[0, -1],
                                self.mid_deep_x1[2]]) / 1e3
            c = ax[iax].pcolormesh(t_sub,
                                   temp_x1,
                                   -deep_creep_slip[2, :-1].reshape(1, -1),
                                   norm=norm, cmap=cmap, shading="flat")
            ax[iax].set_ylabel("Deep Creep\n$x_1$ [km]")
            ax[iax].set_yticks(temp_x1, [f"{temp_x1[0]:.0f}", "$-\\infty$"])
            iax += 1
        # finish figure
        for iax in range(len(ax)):
            for n in range(1, n_cyc_completed):
                ax[iax].axvline(n, c="k", lw=1)
            ax[iax].invert_yaxis()
        ax[-1].set_xlabel("Normalized Time $t/T$")
        fig.colorbar(c, ax=ax, location="right", orientation="vertical", fraction=0.05,
                     label="$(s - t*v_{plate})/s_{full}$" if deficit else "$s/s_{full}$")
        suptitle = "Normalized Fault Patch Slip"
        if deficit:
            suptitle += " Deficit"
        fig.suptitle(suptitle)
        return fig, ax

    def plot_eq_velocities(self, full_state):
        """
        Plot the before and after velocities on all creeping fault patches
        for each distinct earthquake.

        Parameters
        ----------
        full_state : numpy.ndarray
            State matrix as output from :meth:`~run`.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        # get indices of each last earthquake in last cycle
        temp = self.eq_df.astype(bool).drop_duplicates(keep="last")
        time_eq_last = temp.index.values + (self.n_cycles_max - 1) * self.T_eff
        tdiff = np.array([np.min(np.abs(self.t_eval_joint - tlast)) for tlast in time_eq_last])
        if np.any(tdiff > 0):
            warn("Couldn't find exact indices, using time differences of "
                 f"{tdiff * 365.25 * 86400} seconds.")
        ix_eq_last = [np.argmin(np.abs(self.t_eval_joint - tlast)) for tlast in time_eq_last]
        n_eq_found = len(ix_eq_last)
        assert n_eq_found == (self.Ds_0 > 0).sum(), \
            "Couldn't find indices of each last non-zero earthquake in the " \
            "last cycle, check for rounding errors."
        # calculate average slip for plotted earthquakes
        slip_last = self.eq_df.loc[temp.index, :]
        slip_avg = [slip_last.iloc[ieq, np.flatnonzero(temp.iloc[ieq, :])].mean()
                    for ieq in range(n_eq_found)]
        # extract velocities
        vels = full_state[np.r_[self.fault.n_creeping_upper:self.fault.n_state_upper,
                                self.fault.n_state_upper + self.fault.n_creeping_lower:
                                self.fault.n_state_upper + self.fault.n_state_lower],
                          :] / self.v_plate
        # prepare plot
        fig, ax = plt.subplots(nrows=n_eq_found, ncols=1 if self.fault.lower_rheo is None else 2,
                               sharey=True, layout="constrained")
        ax = np.asarray(ax).reshape(n_eq_found, -1)
        # loop over earthquakes
        for irow, ieq in enumerate(ix_eq_last):
            # repeat plot for before and after
            for ioff, label in enumerate(["before", "after"]):
                ax[irow, 0].set_yscale("symlog", linthresh=1)
                ax[irow, 0].plot(self.fault.mid_x1_creeping[:self.fault.n_creeping_upper] / 1e3,
                                 vels[:self.fault.n_creeping_upper, ieq - 1 + ioff],
                                 c=f"C{ioff}", label=label)
                if self.fault.lower_rheo is not None:
                    ax[irow, 1].set_yscale("symlog", linthresh=1)
                    ax[irow, 1].plot(
                        self.fault.mid_x1_creeping[self.fault.n_creeping_upper:] / 1e3,
                        -vels[self.fault.n_creeping_upper:, ieq - 1 + ioff],
                        c=f"C{ioff}", label=label)
        # finish plot
        for irow in range(n_eq_found):
            ax[irow, 0].set_title(f"Upper Interface: $s={slip_avg[irow]:.2g}$ m")
            ax[irow, 0].legend()
            ax[irow, 0].set_xlabel("$x_1$ [km]")
            ax[irow, 0].set_ylabel("$v/v_{plate}$")
            if self.fault.lower_rheo is not None:
                ax[irow, 1].set_title(f"Lower Interface: $s={slip_avg[irow]:.2g}$ m")
                ax[irow, 1].axvline(0, c="k", lw=1)
                ax[irow, 1].axvline(self.fault.x1_lock / 1e3, c="k", lw=1)
                ax[irow, 1].tick_params(labelleft=True)
                ax[irow, 1].legend()
                ax[irow, 1].set_xlabel("$x_1$ [km]")
                ax[irow, 1].set_ylabel("$v/v_{plate}$")
        fig.suptitle("Normalized Earthquake Velocity Changes")
        return fig, ax

    def plot_fault(self):
        """
        Plot the fault.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
        ax.plot(self.fault.end_upper[0, :self.fault.n_locked + 1]/1e3,
                self.fault.end_upper[1, :self.fault.n_locked + 1]/1e3,
                marker="|", markeredgecolor="k",
                label="Locked")
        ax.plot(self.fault.end_upper[0, self.fault.n_locked:]/1e3,
                self.fault.end_upper[1, self.fault.n_locked:]/1e3,
                marker="|", markeredgecolor="k",
                label="Upper Creeping")
        ax.plot(self.fault.end_lower[0, :]/1e3,
                self.fault.end_lower[1, :]/1e3,
                marker="|", markeredgecolor="k",
                label="Lower Creeping")
        ax.plot(self.pts_surf / 1e3, np.zeros_like(self.pts_surf),
                "^", markeredgecolor="none", markerfacecolor="k",
                label="Observers")
        ax.axhline(0, lw=1, c="0.5", zorder=-1)
        ax.legend()
        ax.set_xlabel("$x_1$ [km]")
        ax.set_ylabel("$x_2$ [km]")
        ax.set_title("Fault Mesh and Observer Locations")
        ax.set_aspect("equal")
        return fig, ax

    def plot_slip_phases(self, full_state, post_inter_transition=0.01, normalize=True):
        """
        Plot the cumulative slip on the fault for the three different
        phases (coseismic, early postseismic, and interseismic).

        Only works if there is a single earthquake in the sequence.

        Parameters
        ----------
        full_state : numpy.ndarray
            State matrix as output from :meth:`~run`.
        post_inter_transition : float, optional
            Fraction of the recurrence time that should be considered
            early postseismic and not interseismic.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        # check that the sequence only has one earthquake
        if not self.n_eq == 1:
            raise NotImplementedError("Don't know how to plot slip phases if "
                                      "multiple earthquakes are present in the sequence.")
        # get coseismic slip
        co = np.concatenate([self.eq_df.values.ravel(),
                             self.slip_taper.ravel()])
        # get index of last earthquake in last cycle
        time_eq_last = self.eq_df.index[0] + (self.n_cycles_max - 1) * self.T_eff
        ix_eq_last = (np.flatnonzero(np.isin(self.t_eval_joint, time_eq_last))[0]
                      - self.ix_break_joint[-2])
        # reorganize interseismic slip
        slip = full_state[:self.fault.n_creeping_upper, self.ix_break_joint[-2]:]
        slip_pre = slip[:, :ix_eq_last]
        slip_post = slip[:, ix_eq_last:]
        slip_pre += (slip_post[:, -1] - slip_pre[:, 0]).reshape(-1, 1)
        slip_joint = np.hstack([slip_post, slip_pre])
        slip_joint -= slip_joint[:, 0].reshape(-1, 1)
        # same for time
        t_last = self.t_eval_joint[self.ix_break_joint[-2]:].copy()
        t_last_pre = t_last[:ix_eq_last]
        t_last_post = t_last[ix_eq_last:]
        t_last_pre += t_last_post[-1] - t_last_pre[0]
        t_last_joint = np.concatenate([t_last_post, t_last_pre])
        t_last_joint -= t_last_joint[0]
        # since slip_joint is now already cumulative slip since the earthquake,
        # with the tapered slip removed, we can just read out the early
        # postseismic and rest interseismic cumulative slip distributions
        post = interp1d(t_last_joint, slip_joint)(post_inter_transition * self.T_eff)
        inter = slip_joint[:, -1] - post
        post = np.concatenate([np.zeros(self.fault.n_locked), post])
        inter = np.concatenate([np.zeros(self.fault.n_locked), inter])
        # optionally, normalize by total expected cumulative slip over the entire cycle
        if normalize:
            total_slip = self.T_eff * self.v_plate * 86400 * 365.25
            co /= total_slip
            post /= total_slip
            inter /= total_slip
        # make figure
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(self.fault.mid_x1[:self.fault.n_upper] / 1e3, co, label="Coseismic")
        ax.plot(self.fault.mid_x1[:self.fault.n_upper] / 1e3, post, label="Postseismic")
        ax.plot(self.fault.mid_x1[:self.fault.n_upper] / 1e3, inter, label="Interseismic")
        ax.legend()
        ax.set_xlabel("$x_1$ [km]")
        ax.set_ylabel("Normalized cumulative slip [-]" if normalize
                      else "Cumulative Slip [m]")
        ax.set_title("Slip Phases (Post-/Interseismic cutoff at "
                     f"{post_inter_transition:.1%} " "$T_{rec}$)")
        return fig, ax

    def plot_viscosity(self, full_state, return_viscosities=False):
        """
        Plot the viscosity structure with depth for the steady state, as well as
        for the immediate pre- and coseismic velocities.

        For multiple earthquakes, it will use the minimum preseismic and maximum
        postseismic velocities.

        Parameters
        ----------
        full_state : numpy.ndarray
            State matrix as output from :meth:`~run`.
        return_viscosities : bool, optional
            Also return the preseismic, steady-state, and postseismic viscosities.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        # get indices of each last earthquake in last cycle
        temp = self.eq_df.astype(bool).drop_duplicates(keep="last")
        time_eq_last = temp.index.values + (self.n_cycles_max - 1) * self.T_eff
        tdiff = np.array([np.min(np.abs(self.t_eval_joint - tlast)) for tlast in time_eq_last])
        if np.any(tdiff > 0):
            warn("Couldn't find exact indices, using time differences of "
                 f"{tdiff * 365.25 * 86400} seconds.")
        ix_eq_last = [np.argmin(np.abs(self.t_eval_joint - tlast)) for tlast in time_eq_last]
        n_eq_found = len(ix_eq_last)
        assert n_eq_found == (self.Ds_0 > 0).sum(), \
            "Couldn't find indices of each last non-zero earthquake in the " \
            "last cycle, check for rounding errors."
        # calculate average slip for plotted earthquakes
        slip_last = self.eq_df.loc[temp.index, :]
        slip_avg = [slip_last.iloc[ieq, np.flatnonzero(temp.iloc[ieq, :])].mean()
                    for ieq in range(n_eq_found)]
        # extract preseismic velocities
        vels_pre = np.array([full_state[self.fault.n_creeping_upper:self.fault.n_state_upper,
                                        ix - 1] for ix in ix_eq_last]).T
        vels_post = np.array([full_state[self.fault.n_creeping_upper:self.fault.n_state_upper,
                                         ix] for ix in ix_eq_last]).T
        if isinstance(self.fault.upper_rheo, NonlinearViscous):
            # calculate viscosity profiles
            vis_pre = SubductionSimulation.get_alpha_eff(self.alpha_n_vec.reshape(-1, 1),
                                                         self.n_vec.reshape(-1, 1),
                                                         vels_pre)
            vis_ss = SubductionSimulation.get_alpha_eff(self.alpha_n_vec,
                                                        self.n_vec,
                                                        self.v_plate_eff)
            vis_post = SubductionSimulation.get_alpha_eff(self.alpha_n_vec.reshape(-1, 1),
                                                          self.n_vec.reshape(-1, 1),
                                                          vels_post)
        elif isinstance(self.fault.upper_rheo, RateStateSteadyLogarithmic):
            vis_pre = SubductionSimulation.get_alpha_eff_from_alpha_h(
                self.alpha_h_vec.reshape(-1, 1), vels_pre)
            vis_ss = SubductionSimulation.get_alpha_eff_from_alpha_h(
                self.alpha_h_vec.reshape(-1, 1),  self.v_plate_eff)
            vis_post = SubductionSimulation.get_alpha_eff_from_alpha_h(
                self.alpha_h_vec.reshape(-1, 1),  vels_post)
        else:
            raise NotImplementedError()
        vis_mins = 10**np.floor(np.log10(np.ma.masked_invalid(vis_post*0.999).min(axis=0)))
        vis_maxs = 10**np.ceil(np.log10(np.ma.masked_invalid(vis_pre*1.001).max(axis=0)))
        # make plot
        fig, ax = plt.subplots(ncols=n_eq_found, sharey=True, layout="constrained")
        ax = np.atleast_1d(ax)
        ax[0].set_ylabel("$x_2$ [km]")
        for i in range(n_eq_found):
            ax[i].fill_betweenx([0, self.fault.mid_x2_creeping[1] / 1e3],
                                vis_mins[i], vis_maxs[i], facecolor="0.8", label="Locked")
            ax[i].fill_betweenx(self.fault.mid_x2_creeping[:self.fault.n_creeping_upper] / 1e3,
                                vis_pre[:, i], vis_post[:, i], alpha=0.5, label="Simulated")
            ax[i].plot(vis_ss,
                       self.fault.mid_x2_creeping[:self.fault.n_creeping_upper] / 1e3,
                       label="Plate Rate")
            ax[i].set_xscale("log")
            ax[i].legend(loc="lower left")
            ax[i].set_ylim(self.fault.mid_x2_creeping[self.fault.n_creeping_upper - 1] / 1e3,
                           0)
            ax[i].set_xlim(vis_mins[i], vis_maxs[i])
            ax[i].set_title(f"$s={slip_avg[i]:.2g}$ m")
            ax[i].set_xlabel(r"$\alpha_{eff}$ [Pa * s/m]")
        # finish
        if return_viscosities:
            return fig, ax, vis_pre, vis_ss, vis_post
        else:
            return fig, ax

    def plot_viscosity_timeseries(self, full_state, return_viscosities=False):
        """
        Plot the viscosity timeseries with depth for the entire last cycle.

        Parameters
        ----------
        full_state : numpy.ndarray
            State matrix as output from :meth:`~run`.
        return_viscosities : bool, optional
            Also return the viscosity timeseries.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from cmcrameri import cm
        # check that the sequence only has one earthquake
        if not self.n_eq == 1:
            raise NotImplementedError("Don't know how to plot viscosity timeseries if "
                                      "multiple earthquakes are present in the sequence.")
        # get index of last earthquake in last cycle
        time_eq_last = self.eq_df.index[0] + (self.n_cycles_max - 1) * self.T_eff
        ix_eq_last = (np.flatnonzero(np.isin(self.t_eval_joint, time_eq_last))[0]
                      - self.ix_break_joint[-2])
        # reorganize interseismic velocities
        vels = full_state[self.fault.n_creeping_upper:2*self.fault.n_creeping_upper,
                          self.ix_break_joint[-2]:]
        vels_pre = vels[:, :ix_eq_last]
        vels_post = vels[:, ix_eq_last:]
        vels = np.hstack([vels_post, vels_pre])
        # same for time
        t_last = self.t_eval_joint[self.ix_break_joint[-2]:].copy()
        t_last_pre = t_last[:ix_eq_last]
        t_last_post = t_last[ix_eq_last:]
        t_last_pre += t_last_post[-1] - t_last_pre[0]
        t_last_joint = np.concatenate([t_last_post, t_last_pre])
        t_last_joint -= t_last_joint[0]
        # convert velocities to effective viscosity
        if isinstance(self.fault.upper_rheo, NonlinearViscous):
            vis_ts = SubductionSimulation.get_alpha_eff(self.alpha_n_vec.reshape(-1, 1),
                                                        self.n_vec.reshape(-1, 1),
                                                        vels)
        elif isinstance(self.fault.upper_rheo, RateStateSteadyLogarithmic):
            vis_ts = SubductionSimulation.get_alpha_eff_from_alpha_h(
                self.alpha_h_vec.reshape(-1, 1), vels)
        else:
            raise NotImplementedError()
        # get index of deep transition
        patch_depths = -self.fault.mid_x2_creeping[:self.fault.n_creeping_upper]
        ix_deep = np.argmin(np.abs(patch_depths - self.fault.upper_rheo.deep_transition))
        # subset vels to skip zero-velocity uppermost patch
        vis_ts = vis_ts[1:, :]
        # get percentage of final viscosity
        rel_vis = vis_ts / vis_ts[:, -1][:, None]
        rel_vis_masked = np.ma.MaskedArray(rel_vis, np.diff(rel_vis, axis=1,
                                           prepend=rel_vis[:, 0][:, None]
                                           ) <= 0).filled(np.NaN)
        levels = [0.2, 0.4, 0.6, 0.8]
        rel_vis_iquant = np.concatenate([np.nanargmax(rel_vis_masked > lvl, axis=1, keepdims=True)
                                        for lvl in levels], axis=1)
        # normalize time
        t_sub = t_last_joint / self.T_eff
        # prepare plot
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        # plot velocities
        c = ax.pcolormesh(
            t_sub,
            np.abs(self.fault.end_upper[1, self.fault.n_locked+1:self.fault.n_locked+ix_deep+1]
                   / 1e3),
            vis_ts[:ix_deep-1, :-1],
            norm=LogNorm(vmin=10**np.floor(np.log10(np.median(vis_ts[:ix_deep-1, 0]))),
                         vmax=10**np.ceil(np.log10(np.max(vis_ts[:ix_deep-1, -1])))),
            cmap=cm.batlow, shading="flat")
        for i in range(len(levels)):
            ax.plot(t_sub[rel_vis_iquant[:ix_deep-1, i]],
                    patch_depths[1:ix_deep] / 1e3,
                    color="w")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlim([0, 1])
        # make the y-axis increasing downwards to mimic depth even though we're plotting x1
        ax.invert_yaxis()
        # finish figure
        ax.set_ylabel("Depth $x_2$ [km]")
        ax.set_xlabel("Normalized Time $t/T$")
        fig.colorbar(c, ax=ax, location="right", orientation="vertical", fraction=0.05,
                     label=r"$\alpha_{eff}$")
        fig.suptitle("Effective Viscosity Timeseries")
        # finish
        if return_viscosities:
            return fig, ax, t_sub, vis_ts
        else:
            return fig, ax
