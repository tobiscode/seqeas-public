"""
Kernel functions for 2D subduction in an elastic halfspace.
"""

# general imports
import numpy as np


def Glinedisp(x1, x2, x1midfp, x2midfp, halflen, theta, nu):
    """
    Compute the displacement Green's functions for a shearing and opening
    line crack in a halfspace. Adapted for Python from the ``LDdispHS``
    Matlab function from [davis17]_. References to "C&S" in the source code
    mean [crouchstarfield83]_.

    Parameters
    ----------
    x1 : float, numpy.ndarray
        :math:`x_1` location of evaluation points [m]
    x2 : float, numpy.ndarray
        :math:`x_2` location of evaluation points [m]
    x1midfp : float, numpy.ndarray
        :math:`x_1` location of the midpoint of the fault patches [m]
    x2midfp : float, numpy.ndarray
        :math:`x_2` location of the midpoint of the fault patches [m]
    halflen : float, numpy.ndarray
        Half-length of each fault patch [m]
    theta : float, numpy.ndarray
        The angle [rad] between the fault patch orientation (not the fault
        normal) and the :math:`x_1` axis in the mathematically negative sense,
        i.e. positive clockwise.
    nu : float
        Poisson's ratio [-]

    Returns
    -------
    G : numpy.ndarray
        Displacement kernel for all patches and all evaluation points [m].
        The first half of rows corresponds to the :math:`xx` component,
        the second one to :math:`yy`.

    Notes
    -----

    To compute the displacement at the evaluation points given the shear and normal
    displacements :math:`D_s` and :math:`D_n` in each patch [m], with positive
    values meaning left-lateral shearing and crack opening, respectively,
    perform a matrix computation of the Green's functions with the displacements.

    Example
    -------

    Calculate the stress kernel for three observation points and two fault
    patches:

    >>> import numpy as np
    >>> x1, x2 = np.meshgrid([-1, 0, 1], [0])
    >>> x1, x2 = x1.ravel(), x2.ravel()
    >>> x1midfp, x2midfp = [0.5, 1.5], [-0.5, -1.5]
    >>> halflen, theta, nu = 1/np.sqrt(2), np.pi/4, 0.25
    >>> G = Glinedisp(x1, x2, x1midfp, x2midfp, halflen, theta, nu)

    Calculate the displacement given left-lateral shearing and crack opening in
    both patches:

    >>> Ds, Dn = [1, 1], [0.1, 0.1]
    >>> D = G @ np.concatenate([Ds, Dn])
    >>> print(D.round(3))
    [ 0.289    nan -0.133 -0.094    nan  0.441]

    (The ``nan`` values are where the fault patch breaks the surface.)

    References
    ----------

    .. [davis17] Davis, T. (2017).
       *A new open source boundary element code and its application to geological*
       *deformation: Exploring stress concentrations around voids and the effects*
       *of 3D frictional distributions on fault surfaces* (M.Sc thesis. Aberdeen University)

    .. [crouchstarfield83] Crouch, S. L., & Starfield, A. M. (1983).
       *Boundary element methods in solid mechanics: with applications in rock*
       *mechanics and geological engineering*. Allen & Unwin.
    """
    # format shapes
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(-1, 1)
    x1midfp = np.asarray(x1midfp).reshape(1, -1)
    x2midfp = np.asarray(x2midfp).reshape(1, -1)
    halflen = np.asarray(halflen).reshape(1, -1)
    beta = np.asarray(-theta).reshape(1, -1)

    # input check
    if np.any(x2 > 0) or np.any(x2midfp > 0):
        raise ValueError("x_2 locations need to be zero or less in a halfspace.")

    # Define material constant used in calculating influence coefficients.
    con = 1 / (4*np.pi * (1 - float(nu)))
    sb, cb = np.sin(beta), np.cos(beta)
    s2b, c2b = np.sin(2*beta), np.cos(2*beta)
    s3b, c3b = np.sin(3*beta), np.cos(3*beta)

    # Define array of local coordinates for the observation grid relative to
    # the midpoint and orientation of the ith element.
    # Refer to (Figure 5.6, C&S, p. 91) and eqs. 4.5.1 of C&S, p. 57.
    XB = (x1 - x1midfp)*cb + (x2 - x2midfp)*sb
    YB = -(x1 - x1midfp)*sb + (x2 - x2midfp)*cb

    # Coordinates of the image dislocation (equation 7.4.6 C&S)
    XBi = (x1 - x1midfp)*cb - (x2 + x2midfp)*sb
    YBi = (x1 - x1midfp)*sb + (x2 + x2midfp)*cb

    # Fix roundoff errors in Ybi and Yb from trig function problems
    YB[np.abs(YB) < 1e-10] = 0
    YBi[np.abs(YBi) < 1e-10] = 0

    # Calculate derivatives of the function f(x,y), eq. 5.2.5 of C&S, p. 81.
    # which are used to calculate the displacement and stress components.
    # It is understood that X and Y refer to XB and YB.
    # First abbreviate repeated terms in the derivatives of f(x,y):
    Y2 = YB**2
    XMa, XPa = XB - halflen, XB + halflen
    XMa2, XPa2 = XMa**2, XPa**2
    R1S = XMa2 + Y2
    R2S = XPa2 + Y2

    # Same thing for the image dislocation
    Y2i = YBi**2
    XMai, XPai = XBi - halflen, XBi + halflen
    XMa2i, XPa2i = XMai**2, XPai**2
    R1Si = XMa2i + Y2i
    R2Si = XPa2i + Y2i

    # don't solve for locations where R1S(i) or R2S(i) are zero
    skip_ix = (R1S == 0) | (R2S == 0) | (R1Si == 0) | (R2Si == 0)
    R1S[skip_ix] = np.NaN
    R2S[skip_ix] = np.NaN
    R1Si[skip_ix] = np.NaN
    R2Si[skip_ix] = np.NaN

    R1S2i = R1Si**2
    R2S2i = R2Si**2

    # The following derivatives are eqs. 4.5.5a thru d of C&S, p. 58.
    FF2 = con * (np.log(np.sqrt(R1S)) - np.log(np.sqrt(R2S)))

    # get FF3 directly from engle differences without subindexing
    FF3 = -con * np.angle(np.exp(1j*np.arctan2(YB, XMa))
                          * np.exp(-1j*np.arctan2(YB, XPa)))

    # get the next derivatives
    FF4 = con * (YB / R1S - YB / R2S)
    FF5 = con * (XMa / R1S - XPa / R2S)

    # Calculate intermediate functions Fni for the image dislocation
    FF2i = con * (np.log(np.sqrt(R1Si)) - np.log(np.sqrt(R2Si)))  # Equations 4.5.5 C&S
    FF3i = -con * np.angle(np.exp(1j*np.arctan2(YBi, XMai))
                           * np.exp(-1j*np.arctan2(YBi, XPai)))
    FF4i = con * (YBi / R1Si - YBi / R2Si)
    FF5i = con * (XMai / R1Si - XPai / R2Si)

    # The halfspace examples of eqs. 5.5.3a and b of C&S, p. 91.
    # See Appendix A of: Martel, S.J. and Langley, J.S., 2006. Propagation of
    # normal faults to the surface in basalt, Koae fault system, Hawaii.
    # Journal of Structural Geology, 28(12), pp.2123-2143.
    FF6i = con * ((XMa2i - Y2i) / R1S2i - (XPa2i - Y2i) / R2S2i)
    FF7i = 2*con * YBi * (XMai / R1S2i - XPai / R2S2i)

    # Define material constants used in calculating displacements.
    pr1 = 1 - 2*nu
    pr2 = 2 * (1 - nu)
    pr3 = 3 - 4*nu

    # Calculate the displacement components using eqs. 5.5.4 of C&S, p. 91.
    Gxx_s = -pr1*sb*FF2 + pr2*cb*FF3 + YB * (sb*FF4 - cb*FF5)
    Gxy_n = -pr1*cb*FF2 - pr2*sb*FF3 - YB * (cb*FF4 + sb*FF5)
    Gyx_s = +pr1*cb*FF2 + pr2*sb*FF3 - YB * (cb*FF4 + sb*FF5)
    Gyy_n = -pr1*sb*FF2 + pr2*cb*FF3 - YB * (sb*FF4 - cb*FF5)

    # See equations 7.4.8 and 7.4.9 in Crouch and Starfield
    # Calculate IMAGE AND SUPPLEMENTAL DISPLACEMENT components due to unit SHEAR
    # displacement discontinuity
    Gxxi_s = (pr1 * sb * FF2i - pr2 * cb * FF3i +
              (pr3 * (x2 * s2b - YB * sb) + 2 * x2 * s2b) * FF4i +
              (pr3 * (x2 * c2b - YB * cb) - x2 * (1-2 * c2b)) * FF5i +
              2 * x2 * (x2 * s3b - YB * s2b) * FF6i -
              2 * x2 * (x2 * c3b - YB * c2b) * FF7i)
    Gyxi_s = (-pr1 * cb * FF2i - pr2 * sb * FF3i -
              (pr3 * (x2 * c2b - YB * cb) + x2 * (1-2 * c2b)) * FF4i +
              (pr3 * (x2 * s2b - YB * sb) - 2 * x2 * s2b) * FF5i +
              2 * x2 * (x2 * c3b - YB * c2b) * FF6i +
              2 * x2 * (x2 * s3b - YB * s2b) * FF7i)

    # Calculate IMAGE AND SUPPLEMENTAL DISPLACEMENT components due to unit NORMAL
    # displacement discontinuity
    Gxyi_n = (pr1 * cb * FF2i + pr2 * sb * FF3i -
              (pr3 * (x2 * c2b - YB * cb) - x2) * FF4i +
              pr3 * (x2 * s2b - YB * sb) * FF5i -
              2 * x2 * (x2 * c3b - YB * c2b) * FF6i -
              2 * x2 * (x2 * s3b - YB * s2b) * FF7i)
    Gyyi_n = (pr1 * sb * FF2i - pr2 * cb * FF3i -
              pr3 * (x2 * s2b - YB * sb) * FF4i -
              (pr3 * (x2 * c2b - YB * cb) + x2) * FF5i +
              2 * x2 * (x2 * s3b - YB * s2b) * FF6i -
              2 * x2 * (x2 * c3b - YB * c2b) * FF7i)

    Gxx = Gxx_s + Gxxi_s
    Gxy = -(Gxy_n + Gxyi_n)
    Gyx = Gyx_s + Gyxi_s
    Gyy = -(Gyy_n + Gyyi_n)

    # return stacked displacement kernel
    Gx = np.hstack([Gxx, Gxy])
    Gy = np.hstack([Gyx, Gyy])
    G = np.vstack([Gx, Gy])
    return G


def Klinedisp(x1, x2, x1midfp, x2midfp, halflen, theta, nu, E):
    """
    Compute the stress kernel for a shearing and opening
    line crack in a halfspace. Adapted for Python from the ``LDstressHS``
    Matlab function from [davis17]_. References to "C&S" in the source code
    mean [crouchstarfield83]_.

    Parameters
    ----------
    x1 : float, numpy.ndarray
        :math:`x_1` location of evaluation points [m]
    x2 : float, numpy.ndarray
        :math:`x_2` location of evaluation points [m]
    x1midfp : float, numpy.ndarray
        :math:`x_1` location of the midpoint of the fault patches [m]
    x2midfp : float, numpy.ndarray
        :math:`x_2` location of the midpoint of the fault patches [m]
    halflen : float, numpy.ndarray
        Half-length of each fault patch [m]
    theta : float, numpy.ndarray
        The angle [rad] between the fault patch orientation (not the fault
        normal) and the :math:`x_1` axis in the mathematically negative sense,
        i.e. positive clockwise.
    nu : float
        Poisson's ratio [-]
    E : float
        Young's modulus [Pa]

    Returns
    -------
    K : numpy.ndarray
        Stress kernel for all patches and all evaluation points [m].
        The first third of rows corresponds to the :math:`xx` component,
        the second to :math:`yy`, and the third to :math:`xy`.

    Notes
    -----

    To compute the stress at the evaluation points given the shear and normal
    displacements :math:`D_s` and :math:`D_n` in each patch [m], with positive
    values meaning left-lateral shearing and crack opening, respectively,
    perform a matrix computation of the stress kernel with the displacements.

    Example
    -------

    Calculate the stress kernel for three observation points and two fault
    patches:

    >>> import numpy as np
    >>> x1, x2 = np.meshgrid([-1, 0, 1], [0])
    >>> x1, x2 = x1.ravel(), x2.ravel()
    >>> x1midfp, x2midfp = [0.5, 1.5], [-0.5, -1.5]
    >>> halflen, theta, nu, E = 1/np.sqrt(2), np.pi/4, 0.25, 1
    >>> K = Klinedisp(x1, x2, x1midfp, x2midfp, halflen, theta, nu, E)

    Calculate the stress given left-lateral shearing and crack opening in
    both patches:

    >>> Ds, Dn = [1, 1], [0.1, 0.1]
    >>> S = K @ np.concatenate([Ds, Dn])
    >>> print(S.round(3))
    [ 0.084    nan  0.119 -0.       nan  0.     0.       nan  0.   ]

    (The ``nan`` values are where the fault patch breaks the surface.)

    """
    # format shapes
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(-1, 1)
    x1midfp = np.asarray(x1midfp).reshape(1, -1)
    x2midfp = np.asarray(x2midfp).reshape(1, -1)
    halflen = np.asarray(halflen).reshape(1, -1)
    beta = np.asarray(-theta).reshape(1, -1)

    # input check
    if np.any(x2 > 0) or np.any(x2midfp > 0):
        raise ValueError("x_2 locations need to be zero or less in a halfspace.")

    # The shear modulus, sm, is related to the prescribed elastic constants.
    sm = E / (2 * (1 + float(nu)))
    # Define material constant used in calculating influence coefficients.
    con = 1 / (4*np.pi * (1 - float(nu)))
    cons = 2 * sm
    sb, cb = np.sin(beta), np.cos(beta)
    s2b, c2b = np.sin(2*beta), np.cos(2*beta)
    s3b, c3b = np.sin(3*beta), np.cos(3*beta)
    s4b, c4b = np.sin(4*beta), np.cos(4*beta)

    # Define array of local coordinates for the observation grid relative to
    # the midpoint and orientation of the ith element.
    # Refer to (Figure 5.6, C&S, p. 91) and eqs. 4.5.1 of C&S, p. 57.
    XB = (x1 - x1midfp)*cb + (x2 - x2midfp)*sb
    YB = -(x1 - x1midfp)*sb + (x2 - x2midfp)*cb

    # Coordinates of the image dislocation (equation 7.4.6 C&S)
    XBi = (x1 - x1midfp)*cb - (x2 + x2midfp)*sb
    YBi = (x1 - x1midfp)*sb + (x2 + x2midfp)*cb

    # Fix roundoff errors in Ybi and Yb from trig function problems
    YB[np.abs(YB) < 1e-10] = 0
    YBi[np.abs(YBi) < 1e-10] = 0

    # Calculate derivatives of the function f(x,y), eq. 5.2.5 of C&S, p. 81.
    # which are used to calculate the displacement and stress components.
    # It is understood that X and Y refer to XB and YB.
    # First abbreviate repeated terms in the derivatives of f(x,y):
    Y2 = YB**2
    XMa, XPa = XB - halflen, XB + halflen
    XMa2, XPa2 = XMa**2, XPa**2
    R1S = XMa2 + Y2
    R2S = XPa2 + Y2

    # Same thing for the image dislocation
    Y2i = YBi**2
    XMai, XPai = XBi - halflen, XBi + halflen
    XMa2i, XPa2i = XMai**2, XPai**2
    R1Si = XMa2i + Y2i
    R2Si = XPa2i + Y2i

    # don't solve for locations where R1S(i) or R2S(i) are zero
    skip_ix = (R1S == 0) | (R2S == 0) | (R1Si == 0) | (R2Si == 0)
    R1S[skip_ix] = np.NaN
    R2S[skip_ix] = np.NaN
    R1Si[skip_ix] = np.NaN
    R2Si[skip_ix] = np.NaN

    R1S2, R2S2 = R1S**2, R2S**2
    R1S2i, R2S2i = R1Si**2, R2Si**2

    FF4 = con * (YB / R1S - YB / R2S)
    FF5 = con * (XMa / R1S - XPa / R2S)

    # The following derivatives are eqs. 5.5.3a and b of C&S, p. 91.
    FF6 = con * ((XMa2 - Y2) / R1S2 - (XPa2 - Y2) / R2S2)
    FF7 = 2*con * YB * (XMa / R1S2 - XPa / R2S2)

    FF4i = con * (YBi / R1Si - YBi / R2Si)
    FF5i = con * (XMai / R1Si - XPai / R2Si)

    # The halfspace examples of eqs. 5.5.3a and b of C&S, p. 91.
    # See Appendix A of: Martel, S.J. and Langley, J.S., 2006. Propagation of
    # normal faults to the surface in basalt, Koae fault system, Hawaii.
    # Journal of Structural Geology, 28(12), pp.2123-2143.
    FF6i = con * ((XMa2i - Y2i) / R1S2i - (XPa2i - Y2i) / R2S2i)
    FF7i = 2*con * YBi * (XMai / R1S2i - XPai / R2S2i)

    # don't estimate at invalid points
    HR1 = ((halflen + XBi)**2 + YBi**2)
    HR2 = (YBi**2 + (halflen - XBi)**2)
    skip_ix = (HR1 == 0) | (HR2 == 0)
    HR1[skip_ix] = np.NaN
    HR2[skip_ix] = np.NaN

    # *Tim* I used MATLABs symbolic to find these not eq's A.3 and A.4 of Martel
    # Used Eq.A.1 on variable FF7i (expanded).
    FF8i = (YBi * (1 / HR1**2 - 1 / HR2**2
                   + (2 * (halflen - XBi) * (2 * halflen - 2 * XBi)
                      ) / HR2**3
                   - (2 * (halflen + XBi) * (2 * halflen + 2 * XBi)
                      ) / HR1**3)
            ) / (2 * np.pi * (nu - 1))
    FF9i = (((halflen - XBi) / HR2**2
             + (halflen + XBi) / HR1**2) / (2 * np.pi * (nu - 1))
            - (YBi * ((4 * YBi * (halflen + XBi)) / HR1**3
                      + (4 * YBi * (halflen - XBi)) / HR2**3)
               ) / (2 * np.pi * (nu - 1)))

    # Calculate the stress components using eqs. 5.5.5 of C&S, p. 92.
    Kxx_s = 2*(cb*cb)*FF4 + s2b*FF5 + YB * (c2b*FF6-s2b*FF7)
    Kxx_n = -FF5 + YB * (s2b*FF6 + c2b*FF7)
    Kyy_s = 2*(sb*sb)*FF4 - s2b*FF5 - YB * (c2b*FF6-s2b*FF7)
    Kyy_n = -FF5 - YB * (s2b*FF6 + c2b*FF7)
    Kxy_s = s2b*FF4 - c2b*FF5 + YB * (s2b*FF6+c2b*FF7)
    Kxy_n = -YB * (c2b*FF6 - s2b*FF7)

    #  Calculate IMAGE AND SUPPLEMENTAL STRESS components due to unit SHEAR and
    #  NORMAL displacement discontinuity
    Kxxi_s = (FF4i - 3 * (c2b * FF4i - s2b * FF5i) +
              (2 * x2 * (cb - 3 * c3b) + 3 * YB * c2b) * FF6i +
              (2 * x2 * (sb - 3 * s3b) + 3 * YB * s2b) * FF7i -
              2 * x2 * (x2 * c4b - YB * c3b) * FF8i -
              2 * x2 * (x2 * s4b - YB * s3b) * FF9i)

    Kxxi_n = (FF5i + (2 * x2 * (sb - 2 * s3b) +
              3 * YB * s2b) * FF6i - (2 * x2 * (cb - 2 * c3b) +
              3 * YB * c2b) * FF7i - 2 * x2 * (x2 * s4b - YB * s3b) * FF8i +
              2 * x2 * (x2 * c4b - YB * c3b) * FF9i)

    Kyyi_s = (FF4i - (c2b * FF4i - s2b * FF5i) -
              (4 * x2 * sb * s2b - YB * c2b) * FF6i +
              (4 * x2 * sb * c2b + YB * s2b) * FF7i +
              2 * x2 * (x2 * c4b - YB * c3b) * FF8i +
              2 * x2 * (x2 * s4b - YB * s3b) * FF9i)

    Kyyi_n = (FF5i - (2 * x2 * sb - YB * s2b) * FF6i +
              (2 * x2 * cb - YB * c2b) * FF7i +
              2 * x2 * (x2 * s4b - YB * s3b) * FF8i -
              2 * x2 * (x2 * c4b - YB * c3b) * FF9i)

    Kxyi_s = (s2b * FF4i + c2b * FF5i +
              (2 * x2 * sb * (1 + 4 * c2b) - YB * s2b) * FF6i +
              (2 * x2 * cb * (3 - 4 * c2b) + YB * c2b) * FF7i +
              2 * x2 * (x2 * s4b - YB * s3b) * FF8i -
              2 * x2 * (x2 * c4b - YB * c3b) * FF9i)

    Kxyi_n = ((4 * x2 * sb * s2b + YB * c2b) * FF6i -
              (4 * x2 * sb * c2b - YB * s2b) * FF7i -
              2 * x2 * (x2 * c4b - YB * c3b) * FF8i -
              2 * x2 * (x2 * s4b - YB * s3b) * FF9i)

    Kxx = np.hstack([Kxx_s + Kxxi_s, -Kxx_n - Kxxi_n])
    Kyy = np.hstack([Kyy_s + Kyyi_s, -Kyy_n - Kyyi_n])
    Kxy = np.hstack([Kxy_s + Kxyi_s, -Kxy_n - Kxyi_n])

    # return stacked stress kernel
    K = cons * np.vstack([Kxx, Kyy, Kxy])
    return K
