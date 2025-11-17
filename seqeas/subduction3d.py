"""
Python-only module for the 3D ubduction zone simulations.
"""

# general imports
import json
import configparser
import shapely
import meshcut
import numpy as np
import pandas as pd
import pickle
import cartopy.crs as ccrs
from abc import ABC
from cutde.geometry import compute_efcs_to_tdcs_rotations, strain_to_stress
from cutde.halfspace import strain_matrix, disp_matrix
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, BSpline
from scipy.stats import circmean
from scipy.spatial.distance import cdist
from scipy.special import erf
from warnings import warn
from time import perf_counter
from dataclasses import dataclass, field

# shorthand
SEC_PER_YEAR = 86400 * 365.25
""" Seconds per decimal year of 365.25 days """


def R_ecef2enu(lon, lat):
    """
    Generate the rotation matrix used to express a vector written in ECEF (XYZ)
    coordinates as a vector written in local east, north, up (ENU) coordinates
    at the position defined by geodetic latitude and longitude. See Chapter 4
    and Appendix 4.A in [misraenge2010]_ for details.

    Parameters
    ----------
    lon : float
        Longitude [°] of vector position.
    lat : float
        Latitude [°] of vector position.

    Returns
    -------
        The 3-by-3 rotation matrix.

    References
    ----------

    .. [misraenge2010] Misra, P., & Enge, P. (2010),
       *Global Positioning System: Signals, Measurements, and Performance*,
       Lincoln, Mass: Ganga-Jamuna Press.

    """
    try:
        lon, lat = np.deg2rad(float(lon)), np.deg2rad(float(lat))
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Input longitude & latitude are not convertible to scalars "
            f"(got {lon} and {lat})."
        ).with_traceback(e.__traceback__) from e
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def crossmat(x1: float, x2: float, x3: float) -> np.ndarray:
    """
    Create the matrix for an input vector that if premultiplied with another
    vector will give the same results as cross-multiplying the other vector
    with the input vector.

    Parameters
    ----------
    x1
        First element
    x2
        Second element
    x3
        Third element

    Returns
    -------
        Cross-product matrix
    """
    return np.array([[0, x3, -x2], [-x3, 0, x1], [x2, -x1, 0]])


def correct_custom_utm(coordinates, utmzone, lon0, lat0):
    """
    Correct for a custom UTM Longitude/Latitude offsets by converting input coordinates
    back into Longitude/Latitude, re-adding the offsets, and then converting it back to UTM.

    Parameters
    ----------
    coordinates : numpy.ndarray
        2D ECEF pseudo-UTM coordinates [m]
    utmzone : int
        UTM zone
    lon0 : float, optional
        Longitude offset [°]
    lat0 : float, optional
        Latitude offset [°]

    Returns
    -------
    coordinates_utm : numpy.ndarray
        2D coordinates in standard UTM [m]
    """
    # define coordinate systems
    crs_lla = ccrs.Geodetic()
    crs_utm = ccrs.UTM(zone=utmzone)
    # get reference point in UTM
    coords_ref = np.array(crs_utm.transform_point(lon0, lat0, crs_lla))
    # add reference to current coordinates
    coordinates_utm = coordinates + coords_ref[None, :]
    # # convert to lon/lat
    # coordinates_lla = crs_lla.transform_points(
    #     crs_utm, coordinates[:, 0], coordinates[:, 1])
    # # add offsets
    # if lon0 is not None:
    #     coordinates_lla[:, 0] += lon0
    # if lat0 is not None:
    #     coordinates_lla[:, 1] += lat0
    # # convert back to UTM
    # coordinates_utm = crs_utm.transform_points(
    #     crs_lla, coordinates_lla[:, 0], coordinates_lla[:, 1])[:, :2]
    # done
    return coordinates_utm


def read_tet(filename, delete_invalid=True):
    r"""Read .tet files

    Open a `.tet` file and read the :math:`m` vertices and :math:`n` triangles.
    It checks for duplicate vertices and eliminates them.

    Parameters
    ----------
    filename : str
        Path to the .tet file

    Returns
    -------
    vertices : numpy.ndarray
        :math:`(m, 3)` array of vertices locations
    triangles : numpy.ndarray
        :math:`(n, 3)` array of vertex indices forming a triangle

    Raises
    ------
    RuntimeWarning
        If not all triangles are triangles or if they have extreme aspect ratios.
    """
    # get line number where it switches from vertices to triangles
    with open(filename, mode="rt") as f:
        for m, line in enumerate(f):
            if "TRGL" in line:
                break
    # load vertices
    vertices = np.loadtxt(filename, max_rows=m, usecols=(2, 3, 4), dtype=float)
    # load triangles
    triangles = np.loadtxt(filename, skiprows=m, usecols=(1, 2, 3), dtype=int) - 1
    # check for duplicate vertices
    vertices, triangles = remove_duplicate_vertices(vertices, triangles)
    # check triangles
    invalid_tris = np.flatnonzero([np.unique(tri).size != 3 for tri in triangles])
    if delete_invalid:
        del_ix = []
    if invalid_tris.size > 0:
        warn(
            f"Invalid triangle(s) in file '{filename}' found at line(s)"
            f" {invalid_tris + 1 + m}",
            RuntimeWarning,
        )
        if delete_invalid:
            del_ix.extend(invalid_tris.tolist())
    # check aspect ratios
    ratios = np.empty(triangles.shape[0])
    for i, subtri in enumerate(vertices[triangles]):
        elens = cdist(subtri, subtri).ravel()[[1, 2, 5]]
        ratios[i] = elens.max() / elens.min()
    if np.any(np.log10(ratios) > 1):
        liney_tris = np.flatnonzero(np.log10(ratios) > 1)
        ratioinfo = "\n".join(
            [f"line={i + 1 + m}, index={i}, ratio={ratios[i]}" for i in liney_tris]
        )
        warn(
            f"Found triangles with large aspect ratio in file '{filename}' "
            f"at:\n{ratioinfo}",
            RuntimeWarning,
        )
        if delete_invalid:
            del_ix.extend(liney_tris.tolist())
    # delete triangles if desired, keeping vertices without checking if needed
    if delete_invalid:
        del_ix = list(set(del_ix))
        keep = np.ones(triangles.shape[0], dtype=bool)
        keep[del_ix] = False
        triangles = triangles[keep, :]
    # done
    return vertices, triangles


def write_tet(filename, vertices, triangles):
    r"""Write .tet files

    Write :math:`m` vertices and :math:`n` triangles to a `.tet` file.

    Parameters
    ----------
    filename : str
        Path to the .tet file
    vertices : numpy.ndarray
        :math:`(m, 3)` array of vertices locations
    triangles : numpy.ndarray
        :math:`(n, 3)` array of vertex indices forming a triangle
    """
    # check
    assert triangles.ndim == vertices.ndim == 2
    assert np.max(triangles) == vertices.shape[0] - 1
    # get line number where it switches from vertices to triangles
    with open(filename, mode="wt") as f:
        for i, v in enumerate(vertices):
            f.write(f"VRTX {i + 1} {v[0]:f} {v[1]:f} {v[2]:f}\n")
        for i, t in enumerate(triangles + 1):
            f.write(f"TRGL {t[0]:d} {t[1]:d} {t[2]:d}\n")
    # done
    return


def force_normals(vertices, triangles, normals_up):
    """
    Enforce a direction for all the normals in the mesh.

    Parameters
    ----------
    vertices : numpy.ndarray
        :math:`(m, 3)` array of vertices locations
    triangles : numpy.ndarray
        :math:`(n, 3)` array of vertex indices forming a triangle
    normals_up : bool
        If ``True``, force the normals to point upwards, if ``False``, downwards

    Returns
    -------
    vertices : numpy.ndarray
        :math:`(m, 3)` array of vertices locations
    triangles : numpy.ndarray
        :math:`(n, 3)` array of vertex indices forming a triangle
    """
    # get current orientations
    signs = np.sign(
        [
            np.cross(
                vertices[triangles[i, 1]] - vertices[triangles[i, 0]],
                vertices[triangles[i, 2]] - vertices[triangles[i, 0]],
            )
            for i in range(triangles.shape[0])
        ]
    )
    # find the ones that need to be flipped
    if normals_up:
        change_dirs = signs[:, 2] < 0
    else:
        change_dirs = signs[:, 2] > 0
    # flip by changing the order of the second and third element
    if change_dirs.sum() > 0:
        triangles[np.ix_(change_dirs, [1, 2])] = triangles[np.ix_(change_dirs, [2, 1])]
    # done
    return vertices, triangles


def combine_tet(*args, return_sources=False):
    """
    Combine multiple .tet files into a single combination of :math:`m` vertices
    and :math:`n` triangles. It checks for duplicate vertices and eliminates them.
    Parameters
    ----------
    *args : tuple
        Paths to `.tet` mesh files.
    return_sources : bool, optional
        If ``True``, return a nested list matching each triangle index to a
        source file

    Returns
    -------
    vertices : numpy.ndarray
        :math:`(m, 3)` array of vertices locations
    triangles : numpy.ndarray
        :math:`(n, 3)` array of vertex indices forming a triangle
    match_sources : list
        Nested list that contains the indices of the triangles belonging to
        each individual source mesh file (i.e., the ith list in `match_sources`
        contains all the indices for the ith file).
    """

    # quick return if it's just a single mesh
    if len(args) == 1:
        vertices, triangles = read_tet(args[0])
        if return_sources:
            match_sources = [np.arange(triangles.shape[0])]

    # full workflow
    else:

        # load meshes
        list_verts_tris = [read_tet(mf) for mf in args]
        list_num_vertices = [vt[0].shape[0] for vt in list_verts_tris]
        list_num_patches = [vt[1].shape[0] for vt in list_verts_tris]
        num_vertices = sum(list_num_vertices)
        num_patches = sum(list_num_patches)

        # create empty arrays
        vertices = np.empty((num_vertices, 3))
        triangles = np.empty((num_patches, 3), dtype=int)

        # fill them sequentially
        # this also make the triangle indices continuous
        iv, it = 0, 0
        for vt, nv, nt in zip(list_verts_tris, list_num_vertices, list_num_patches):
            vertices[iv : iv + nv, :] = vt[0]
            triangles[it : it + nt, :] = vt[1] + iv
            iv += nv
            it += nt
        assert num_vertices == iv, f"{num_vertices} != {iv}"
        assert num_patches == it, f"{num_patches} != {it}"
        assert triangles.max() < num_vertices, f"{triangles.max()} >= {num_vertices}!"

        # check for duplicate vertices
        vertices, triangles, old_indices = remove_duplicate_vertices(
            vertices, triangles, return_old_indices=True
        )

        # if desired, create list of source indices
        if return_sources:
            # create raw list
            num_sources = len(list_verts_tris)
            match_sources = [
                np.arange(sum(list_num_patches[:i]), sum(list_num_patches[: i + 1]))
                for i in range(num_sources)
            ]
            # update it from the deduplicated set
            if old_indices is not None:
                for inew, iold in enumerate(old_indices):
                    for i in range(num_sources):
                        match_sources[i][match_sources[i] == iold] = inew

    # done
    if return_sources:
        return vertices, triangles, match_sources
    else:
        return vertices, triangles


def remove_duplicate_vertices(vertices, triangles, return_old_indices=False):
    """
    Given a set of vertices and triangles, remove any potentially
    present duplicate vertices.

    Parameters
    ----------
    vertices : numpy.ndarray
        :math:`(m, 3)` array of vertices locations
    triangles : numpy.ndarray
        :math:`(n, 3)` array of vertex indices forming a triangle
    return_old_indices : bool, optional
        If ``True``, return a list that matches old vertex indices to
        the new ones.

    Returns
    -------
    vertices : numpy.ndarray
        :math:`(m, 3)` array of vertices locations without duplicates
    triangles : numpy.ndarray
        :math:`(n, 3)` array of vertex indices forming a triangle, reindexed
    old_indices : numpy.ndarray
        Array containing all the old indices, indexed to the new ones.
    """

    # check if there are duplicates
    _, unique_indices, unique_counts = np.unique(
        vertices, return_index=True, return_counts=True, axis=0
    )

    # there are duplicate values
    if np.any(unique_counts > 1):
        # sort the unique indices
        isort = np.argsort(unique_indices)
        unique_indices = unique_indices[isort]
        unique_counts = unique_counts[isort]
        # keep old state for testing
        old_patches = vertices[triangles]
        # release ties to previous variables
        vertices = vertices.copy()
        triangles = triangles.copy()
        # match unique to duplicate vertex indices
        match_indices = {
            i: [
                j
                for j in np.flatnonzero(np.all(vertices == vertices[i, :], axis=1))
                if j not in unique_indices
            ]
            for ii, i in enumerate(unique_indices)
            if unique_counts[ii] > 1
        }
        duplicate_indices = [
            i for i in range(vertices.shape[0]) if i not in unique_indices
        ]
        # double-check that every duplicate index is taken care of
        assert all(
            [
                di in [mi for sublist in match_indices.values() for mi in sublist]
                for di in duplicate_indices
            ]
        )
        # loop over every duplicate element
        for iuniq, idups in match_indices.items():
            # replace the duplicate index with the unique one
            triangles[np.isin(triangles, idups)] = iuniq
        # re-index all the vertices
        old_indices = np.arange(vertices.shape[0])[unique_indices]
        for inew, iold in enumerate(old_indices):
            triangles[triangles == iold] = inew
        # delete all duplicate rows
        vertices = vertices[unique_indices, :]
        # check that everything has worked
        new_patches = vertices[triangles]
        assert np.all(old_patches == new_patches)
        assert np.unique(vertices, axis=0).shape[0] == vertices.shape[0]
    elif return_old_indices:
        old_indices = None

    # done
    if return_old_indices:
        return vertices, triangles, old_indices
    else:
        return vertices, triangles


def extend_mesh(
    vertices_in,
    triangles_in,
    dist_inf=1e7,
    directions=[True, True, True, True],
    use_normals=[False, False, False, False],
    normals_in=None,
):
    """
    Extend the mesh in the x and y directions along the respective other axis and
    constant z value. Includes corners.

    Parameters
    ----------
    vertices_in : numpy.ndarray
        :math:`(m, 3)` array of input vertices locations
    triangles_in : numpy.ndarray
        :math:`(n, 3)` array of input vertex indices forming a triangle
    dist_inf : float, optional
        The distance to be considered infinite.
    directions : list, optional
        List of boolean flags indicating whether to extend a certain direction
        or not, ordered as [pos_x, pos_y, neg_x, neg_y].
        By default, all sides are extended. Set to ``False`` sides that should
        be excluded. Corners are only included if both directions are included.
    use_normals : list, optional
        List of boolean flags indicating whether to use the triangle's plane angle
        when extending or not, ordered the same as `directions`.
        By default, all sides are extended at constant :math:`z` value.
    normals_in : numpy.ndarray, optional
        :math:`(n, 3)` array of normal vectors of each triangle

    Returns
    -------
    vertices_out : numpy.ndarray
        :math:`(p, 3)` array of output vertices locations
    triangles_out : dict
        :math:`(q, 3)` array of output vertex indices
    """

    # input checks
    assert dist_inf > 0
    if any(use_normals):
        assert isinstance(normals_in, np.ndarray)

    # build list of directions
    check_axes = [0, 1, 0, 1]
    check_dirs = [1, 1, -1, -1]
    off_axes = [(a + 1) % 2 for a in check_axes]
    off_dirs = [1, -1, -1, 1]

    # initialize output
    n_off = vertices_in.shape[0]
    vertices_out = []
    triangles_out = []
    corner_help = []
    if any(use_normals):
        centroids = np.mean(vertices_in[triangles_in], axis=1)

    # for each side and direction, create vertices and triangles
    for b, a, d, oa, od, n in zip(
        directions, check_axes, check_dirs, off_axes, off_dirs, use_normals
    ):
        if not b:
            corner_help.append([])
            continue
        # get all vertex indices at border
        i_max = np.argwhere(
            np.isclose(vertices_in[:, a], d * np.max(d * vertices_in[:, a]))
        ).ravel()
        n_max = i_max.size
        if n_max == 1:
            # the side is not a clear boundary, but is instead defined
            # by an intersection - let's hope they have the same depth
            i_max = np.argwhere(
                np.isclose(vertices_in[:, 2], vertices_in[i_max, 2])
            ).ravel()
            n_max = i_max.size
            # if it's still just a single value, we can't find the border,
            # and therefore can't extend it - let's hope it was the free surface
            if n_max == 1:
                warn(f"Can't extend axis {a} in the {d:+d} direction.")
                corner_help.append([])
                continue
        # sort both by increasing off-axes value
        sort_i = np.argsort(vertices_in[i_max, oa])[::od]
        i_max = i_max[sort_i]
        if n:
            # translate along plane angle
            # find the triangles with two vertices on the boundary
            tri_contains_border = np.isin(triangles_in, i_max)
            i_tri_at_border = tri_contains_border.sum(axis=1) == 2
            assert i_tri_at_border.sum() == n_max - 1
            # sort index by rising centroid off-axes
            sort_i_tri = np.argsort(centroids[i_tri_at_border, oa])[::od]
            i_tri_at_border = np.flatnonzero(i_tri_at_border)[sort_i_tri]
            # get the respective normal vectors
            normal_at_border = normals_in[i_tri_at_border, :].copy()
            # make sure the normal shows up
            i_normal_down = np.flatnonzero(normal_at_border[:, 2] < 0)
            normal_at_border[i_normal_down, :] *= -1
            # get the angles of the normals w.r.t. the vertical
            ang = np.arccos(
                normal_at_border[:, 2] / np.linalg.norm(normal_at_border, axis=1)
            )
            # take the average angle
            ang = circmean(ang)
            sign = np.sign(np.mean(normal_at_border[:, a]))
            # create final border points by translating the maximum border
            # vertex by the average angle
            new_vert = vertices_in[i_max, :].copy()
            i_border_max = np.argmax(d * new_vert[:, a])
            new_vert[:, [a, 0]] = new_vert[i_border_max, [a, 0]][None, :]
            tilted_dir = np.zeros((1, 3))
            tilted_dir[0, 2] = -np.sin(ang) * dist_inf * sign * d
            tilted_dir[0, a] = np.cos(ang) * dist_inf * d
            new_vert += tilted_dir
        else:
            # just translation along z
            new_vert = vertices_in[i_max, :].copy()
            new_vert[:, a] += d * dist_inf
        # create new triangles between (1, 2) original and (2, 1) new vertices
        i_new = np.arange(n_off, n_off + i_max.size)
        new_tri = np.concatenate(
            [
                [[i_max[i], i_max[i + 1], i_new[i]] for i in range(n_max - 1)],
                [[i_max[i + 1], i_new[i + 1], i_new[i]] for i in range(n_max - 1)],
            ]
        )
        # append
        vertices_out.append(new_vert)
        triangles_out.append(new_tri)
        # save indices of first and last old and new vertices for corner calculation
        corner_help.append([i_max[0], i_max[-1], i_new[0], i_new[-1]])
        # progress
        n_off += n_max

    # concatenate new vertices and indices
    temp_vertices_out = np.concatenate(
        [vertices_in, np.concatenate(vertices_out, axis=0)], axis=0
    )

    # check for corners to add
    for i1, b in enumerate(directions):
        i2 = (i1 + 1) % 4
        # we only have corners if both the current and next directions are extended
        if (not b) or not (directions[i2]):
            continue
        # get the axes and directions of both cornering sides
        a1, ch1 = check_axes[i1], corner_help[i1]
        a2, ch2 = check_axes[i2], corner_help[i2]
        if not (ch1 and ch2):
            continue
        # get the index of the corner vertex
        i_max1, i_new1 = ch1[1], ch1[3]
        i_max2, i_new2 = ch2[0], ch2[2]
        assert i_max1 == i_max2, f"{i1, b}: {i_max1} != {i_max2}"
        # translate to get the new almost-infinitely-far vertex
        new_vert = np.zeros((1, 3))
        new_vert[0, a1] = temp_vertices_out[i_new1, a1]
        new_vert[0, a2] = temp_vertices_out[i_new2, a2]
        if np.logical_xor(use_normals[i1], use_normals[i2]):
            # use the side that uses normals as the indicator for depth
            if use_normals[i1]:
                new_vert[0, 2] = temp_vertices_out[i_new1, 2]
            else:
                new_vert[0, 2] = temp_vertices_out[i_new2, 2]
        else:
            # average the two corner depths
            new_vert[0, 2] = (
                temp_vertices_out[i_new1, 2] + temp_vertices_out[i_new2, 2]
            ) / 2
        # create new triangles
        i_new = n_off
        new_tri = np.array([[i_max1, i_new1, i_new], [i_max1, i_new, i_new2]])
        # append
        vertices_out.append(new_vert)
        triangles_out.append(new_tri)
        # progress
        n_off += 1

    # concatenate new vertices and indices
    vertices_out = np.concatenate([vertices_in, np.concatenate(vertices_out)], axis=0)
    triangles_out = np.concatenate(
        [triangles_in, np.concatenate(triangles_out)], axis=0
    )

    # done
    return vertices_out, triangles_out


def split_mesh(vertices, triangles, patches=None, max_distance=None, max_depth=None):
    """
    Split the mesh into an inside and outside region based on distance to
    other patches and/or depth.

    Resets the triangle indices for the two returned meshes.
    It also does not check whether the vertices and triangles are
    outside of the patches.

    Parameters
    ----------
    vertices : numpy.ndarray
        :math:`(m, 3)` array of input vertices locations
    triangles : numpy.ndarray
        :math:`(n, 3)` array of input vertex indices forming a triangle
    patches : numpy.ndarray, optional
        :math:`(n, 3, 3)` array of patches to avoid
    max_distance : float, optional
        Maximum distance of the mash centroids away from the ``patches`` vertices
        to keep in the inside mesh.
    max_depth : float, optional
        Only keep mesh centroids in the inside mesh if they are above ``max_depth``.

    Returns
    -------
    vertices_inside : numpy.ndarray
        Inside vertices
    vertices_outside : numpy.ndarray
        Outside vertices
    triangles_inside : numpy.ndarray
        Inside triangles
    triangles_outside : numpy.ndarray
        Outside triangles
    """
    # check if we're actually doing something
    assert (max_depth is not None) or (
        (max_distance is not None) and (patches is not None)
    )
    # initialize list of vertex indices to split off
    ix_v_split = []
    # find indices of vertices below below max_depth
    if max_depth is not None:
        ix_v_split.extend(np.flatnonzero(vertices[:, 2] < -max_depth).tolist())
    # find indices more than horizontal max_distance away from patches
    if max_distance is not None:
        # get unique vertex coordinates of asperities
        asp_vert_unique = np.unique(patches.reshape(-1, 3), axis=0)
        # get distance between all asperity vertices and all other vertices
        dists = cdist(vertices, asp_vert_unique)
        # get minimum distance to asperities
        min_dists = np.min(dists, axis=1)
        # get indices where that minimum is larger than max_distance
        ix_v_split.extend(np.flatnonzero(min_dists > max_distance).tolist())
    # get indices of triangles containing any of the split-off vertices
    ix_t_split = np.any(np.isin(triangles, np.unique(ix_v_split)), axis=1)
    ix_t_keep = ~ix_t_split
    # get unique vertex indices to split off or kept (there is overlap)
    ix_v_split = np.unique(triangles[ix_t_split, :].ravel())
    ix_v_keep = np.unique(triangles[ix_t_keep, :].ravel())
    # split vertices
    vertices_outside = vertices[ix_v_split, :].copy()
    vertices_inside = vertices[ix_v_keep, :].copy()
    # split triangles
    triangles_outside = triangles[ix_t_split, :].copy()
    triangles_inside = triangles[ix_t_keep, :].copy()
    # adjust vertex indices
    ix_old = np.arange(vertices.shape[0])
    ix_old_inside = ix_old[ix_v_keep]
    ix_old_outside = ix_old[ix_v_split]
    for inew, iold in enumerate(ix_old_inside):
        triangles_inside[triangles_inside == iold] = inew
    for inew, iold in enumerate(ix_old_outside):
        triangles_outside[triangles_outside == iold] = inew
    # done
    return vertices_inside, vertices_outside, triangles_inside, triangles_outside


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
    alpha_h_vec : numpy.ndarray
        Rate-and-state parameter :math:`(a - b) * \sigma_E`

    Returns
    -------
    dzetadt : numpy.ndarray
        Velocity derivative :math:`\frac{d \zeta}{dt}` [1/s] in logarithmic space.
    """
    return dtaudt / alpha_h_vec


def get_new_vel_rdlog(zeta_minus, delta_tau, alpha_h_vec):
    r"""
    Calculate the instantaneous velocity change (in logarithmic space) due to an
    instantaneous stress change to the fault patches. We can kickstart the
    derivation from the expression in ``RateStateSteadyLinear.get_new_vel``:

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


def flat_ode_rdlog(
    t,
    state,
    n_patches,
    v_plate_vec,
    K_int,
    K_ext_v_plate,
    v_0,
    alpha_h_vec,
    mu_over_2vs,
):
    r"""
    Flattened ODE derivative function for a subduction fault with
    rate-dependent rheology in the upper plate interface, and an imposed
    constant plate velocity at the lower interface (which can be ignored).

    Parameters
    ----------
    t : float
        Current time (needs to be in function call for solve_ivp).
    state : numpy.ndarray
        1D array with the current state of the creeping fault patches,
        containing (in order) the cumulative slip and current velocity,
        and flattened so that its total length is `n_patches * 2 * 2`.
    n_patches : float
        Number of patches in the state array.
    v_plate_vec : numpy.ndarray
        2D plate velocities.
    K_int : numpy.ndarray
        4D tensor with the stress kernel mapping creeping patches to themselves.
    K_ext_v_plate : numpy.ndarray
        Stressing rate induced by the locked asperities.
    v_0 : float
        Reference velocity [m/s]
    alpha_h_vec : numpy.ndarray
        1D column rate-and-state parameter :math:`(a - b) * \sigma_E`
    mu_over_2vs : float
        Radiation damping factor

    Returns
    -------
    dstatedt : numpy.ndarray
        1D array with the state derivative.
    """
    # reshape input state
    state2d = state.reshape(2 * n_patches, 2)
    # extract total velocities
    zeta = state2d[n_patches:, :]
    v = v_0 * np.exp(zeta)
    # get shear strain rate
    rad_damp_factor = mu_over_2vs * v / alpha_h_vec
    dtaudt = (
        np.tensordot(K_int, v - v_plate_vec, axes=[(3, 2), (1, 0)]) - K_ext_v_plate
    ) / (1 + rad_damp_factor)
    # get ODE
    dstatedt = np.concatenate((v, dzetadt_rdlog(dtaudt, alpha_h_vec)))
    # return
    return dstatedt.ravel()


def flat_ode_rdreg(
    t,
    state,
    n_patches,
    v_plate_vec,
    K_int,
    K_ext_v_plate,
    rho,
    alpha_h_vec,
    mu_over_2vs,
):
    r"""
    Flattened ODE derivative function for a subduction fault with
    regularized rate-dependent rheology in the upper plate interface, and an
    imposed constant plate velocity at the lower interface (which can be ignored).

    Parameters
    ----------
    t : float
        Current time (needs to be in function call for solve_ivp).
    state : numpy.ndarray
        1D array with the current state of the creeping fault patches,
        containing (in order) the cumulative slip and current velocity,
        and flattened so that its total length is `n_patches * 2 * 2`.
    n_patches : float
        Number of patches in the state array.
    v_plate_vec : numpy.ndarray
        2D plate velocities.
    K_int : numpy.ndarray
        4D tensor with the stress kernel mapping creeping patches to themselves.
    K_ext_v_plate : numpy.ndarray
        Stressing rate induced by the locked asperities.
    rho : float
        :math:`f_0 / (a-b)` [-]
    alpha_h_vec : numpy.ndarray
        1D column rate-and-state parameter :math:`(a - b) * \sigma_E`
    mu_over_2vs : float
        Radiation damping factor

    Returns
    -------
    dstatedt : numpy.ndarray
        1D array with the state derivative.
    """
    # reshape input state
    state2d = state.reshape(2 * n_patches, 2)
    # extract variables
    # slip = state2d[:n_patches, :]
    traction = state2d[n_patches:, :]
    # get current velocity
    vel = v_plate_vec * np.exp(-rho) * np.sinh(traction / alpha_h_vec)
    # get traction rate
    dtractiondt = (
        np.tensordot(K_int, vel - v_plate_vec, axes=[(3, 2), (1, 0)]) - K_ext_v_plate
    )
    # form state derivative
    dstatedt = np.concatenate((vel, dtractiondt))
    # return
    return dstatedt.ravel()


def flat_run_rdlog(
    t_eval,
    i_break,
    i_eq,
    K_int,
    K_ext_v_plate,
    v_plate_vec,
    v_init,
    delta_tau_bounded,
    delta_tau_bounded_nonuni,
    v_0,
    alpha_h_vec,
    mu_over_2vs,
    v_max,
    atol,
    rtol,
    spinup_atol,
    spinup_rtol,
    verbose,
):
    r"""
    Run the rate-dependent simulation.

    Parameters
    ----------
    t_eval : numpy.ndarray
        Evaluation times [s]
    i_break : numpy.ndarray
        Integer indices of cycle breaks [-]
    i_eq : numpy.ndarray
        Integer indices of earthquakes within sequence [-]
    K_int : numpy.ndarray
        Internal stress kernel [Pa/m]
    K_ext_v_plate : numpy.ndarray
        Stressing rate induced by the locked asperities.
    v_plate_vec : numpy.ndarray
        Plate velocity for all creeping patches [m/s]
    v_init : numpy.ndarray
        Initial velocity of the fault patches, in the dimensions of the rheology
    delta_tau_bounded : numpy.ndarray
        Bounded coseismic stress change [Pa]
    v_0 : float
        Reference velocity [m/s]
    alpha_h_vec : numpy.ndarray
        Upper interface rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa]
    mu_over_2vs : float
        Radiation damping factor :math:`\mu / 2 v_s`, where :math:`\mu` is the shear
        modulus [Pa] and :math:`v_s` is the shear wave velocity [m/s]
    v_max : float
        Maximum velocity that a patch can have during the integration [m/s], or None
        if not limit.
    atol : float
        Absolute tolerance of ODE integrator
    rtol : float
        Relative tolerance of ODE integrator
    spinup_atol : float
        Absolute tolerance of spinup controller
    spinup_rtol : float
        Relative tolerance of spinup controller
    verbose : bool
        Print integration progress.

    Returns
    -------
    sim_state : numpy.ndarray
        Full state variable at the end of the integration.
    """

    # initialize parameters
    n_eval = t_eval.size
    n_slips = delta_tau_bounded.shape[0]
    n_patches = K_int.shape[0]
    assert np.all(np.diff(t_eval) > 0)
    assert (
        n_patches
        == K_ext_v_plate.shape[0]
        == v_plate_vec.shape[0]
        == v_init.shape[0]
        == alpha_h_vec.shape[0]
    )

    # initialize arrays
    signs = np.sign(v_plate_vec)
    assert np.all(signs == np.sign(v_init)) and np.all(signs > 0)
    v_init = np.abs(v_init)
    s_minus_upper = np.zeros((n_patches, 2))
    zeta_minus_upper = np.log(v_init / v_0)
    sim_state = np.full((2 * n_patches, 2, n_eval), np.nan)
    state_plus = np.concatenate((s_minus_upper, zeta_minus_upper))

    # make flat ODE function arguments
    args = (n_patches, v_plate_vec, K_int, K_ext_v_plate, v_0, alpha_h_vec, mu_over_2vs)

    # integrate
    spun_up = 0
    i_slip = 0
    steps = np.sort(np.concatenate((i_eq, i_break)))
    i = 0
    if delta_tau_bounded_nonuni is not None:
        # need to get the indices of the last n_slips_obs events
        n_slips_obs = delta_tau_bounded_nonuni.shape[0]
        i_eq_nonuni = i_eq[-n_slips_obs:].tolist()
    else:
        i_eq_nonuni = []
    if verbose:
        print("0", end="", flush=True)
        ii = 1
    while i < steps.size - 1:
        # get indices
        ji, jf = steps[i], steps[i + 1]
        ti, tf = t_eval[ji], t_eval[jf]
        # call integrator
        sol = solve_ivp(
            flat_ode_rdlog,
            t_span=[ti, tf],
            y0=state_plus.ravel(),
            t_eval=t_eval[ji : jf + 1],
            method="RK45",
            args=args,
            atol=atol,
            rtol=rtol,
        )
        success = sol.success
        if success:
            sol = sol.y.reshape(2 * n_patches, 2, -1)
        else:
            raise RuntimeError("Integrator failed.")
        # save state to output array
        sim_state[:, :, ji : jf + 1] = sol
        # can already stop here if this is the last interval
        if i == steps.size - 2:
            break
        # at the end of a full cycle, check the early stopping criteria
        if (not spun_up) and (i > n_slips) and (jf in i_break):
            if verbose:
                print(f"{ii}", end="", flush=True)
                ii += 1
            old_full_state = sim_state[
                :, :, steps[i - 2 * n_slips - 1] : steps[i - n_slips]
            ]
            new_full_state = sim_state[:, :, steps[i - n_slips] : steps[i + 1]]
            old_zeta_upper = old_full_state[n_patches:, :, -1]
            new_zeta_upper = new_full_state[n_patches:, :, -1]
            lhs_upper = np.abs(new_zeta_upper - old_zeta_upper)
            rhs_upper = spinup_atol + spinup_rtol * np.fmax(
                np.abs(new_zeta_upper), np.abs(old_zeta_upper)
            )
            stop_now = np.all(lhs_upper <= rhs_upper)
            if stop_now:
                spun_up = jf
                # advance i to the last cycle (don't forget the general advance later)
                if verbose:
                    print(">", end="", flush=True)
                i = steps.size - n_slips - 3
        elif (not spun_up) and (jf in i_break) and verbose:
            print(f"{ii}", end="", flush=True)
            ii += 1
        elif spun_up and (jf in i_break):
            break
        # apply step change only if there is one
        if jf in i_eq:
            if verbose:
                print(".", end="", flush=True)
            state_upper = sol[:, :, -1]
            s_minus_upper = state_upper[:n_patches, :]
            zeta_minus_upper = state_upper[n_patches:, :]
            s_plus_upper = s_minus_upper.copy()
            if spun_up and (jf in i_eq_nonuni):
                i_slip_nonuni = i_eq_nonuni.index(jf)
                zeta_plus_upper = get_new_vel_rdlog(
                    zeta_minus_upper,
                    delta_tau_bounded_nonuni[i_slip_nonuni, :, :],
                    alpha_h_vec,
                )
            else:
                zeta_plus_upper = get_new_vel_rdlog(
                    zeta_minus_upper, delta_tau_bounded[i_slip, :, :], alpha_h_vec
                )
            if v_max is not None:
                v_plus_upper = v_0 * np.exp(zeta_plus_upper)
                v_plus_upper_norm = np.linalg.norm(v_plus_upper, axis=1)
                if np.any(v_plus_upper_norm > v_max):
                    v_plus_upper *= (
                        np.clip(v_plus_upper_norm, a_min=None, a_max=v_max)
                        / v_plus_upper_norm
                    )[:, None]
                    zeta_plus_upper = np.log(v_plus_upper / v_0)
            state_plus = np.concatenate((s_plus_upper, zeta_plus_upper))
            i_slip = (i_slip + 1) % n_slips
        else:
            state_plus = sol[:, :, -1]
        # advance
        i += 1

    # warn if we never spun up
    if not spun_up:
        print(f"Simulation did not spin up after {len(i_break) - 1} cycles!")

    # convert from log to linear velocity
    sim_state[n_patches:, :, :] = v_0 * np.exp(sim_state[n_patches:, :, :])

    # done
    return sim_state


def flat_run_rdreg(
    t_eval,
    i_break,
    i_eq,
    K_int,
    K_ext_v_plate,
    v_plate_vec,
    v_init,
    delta_tau_bounded,
    delta_tau_bounded_nonuni,
    slip_taper_vec,
    slip_taper_vec_nonuni,
    rho,
    alpha_h_vec,
    mu_over_2vs,
    v_max,
    atol,
    rtol,
    spinup_atol,
    spinup_rtol,
    verbose,
):
    r"""
    Run the regularized rate-dependent simulation.

    Parameters
    ----------
    t_eval : numpy.ndarray
        Evaluation times [s]
    i_break : numpy.ndarray
        Integer indices of cycle breaks [-]
    i_eq : numpy.ndarray
        Integer indices of earthquakes within sequence [-]
    K_int : numpy.ndarray
        Internal stress kernel [Pa/m]
    K_ext_v_plate : numpy.ndarray
        Stressing rate induced by the locked asperities.
    v_plate_vec : numpy.ndarray
        Plate velocity for all creeping patches [m/s]
    v_init : numpy.ndarray
        Initial velocity of the fault patches, in the dimensions of the rheology
    delta_tau_bounded : numpy.ndarray
        Bounded coseismic stress change [Pa]
    delta_tau_bounded_nonuni : numpy.ndarray
        Bounded coseismic stress change for nonuniform earthquakes [Pa]
    slip_taper_vec : numpy.ndarray
        Tapered slip [m]
    slip_taper_vec_nonuni : numpy.ndarray
        Tapered slip for nonuniform earthquakes [m]
    rho : float
        :math:`f_0 / (a-b)` [-]
    alpha_h_vec : numpy.ndarray
        Upper interface rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa]
    mu_over_2vs : float
        Radiation damping factor :math:`\mu / 2 v_s`, where :math:`\mu` is the shear
        modulus [Pa] and :math:`v_s` is the shear wave velocity [m/s]
    v_max : float
        Maximum velocity that a patch can have during the integration [m/s], or None
        if not limit.
    atol : float
        Absolute tolerance of ODE integrator
    rtol : float
        Relative tolerance of ODE integrator
    spinup_atol : float
        Absolute tolerance of spinup controller
    spinup_rtol : float
        Relative tolerance of spinup controller
    verbose : bool
        Print integration progress.

    Returns
    -------
    sim_state : numpy.ndarray
        Full state variable at the end of the integration.
    """

    # initialize parameters
    n_eval = t_eval.size
    n_slips = delta_tau_bounded.shape[0]
    n_patches = K_int.shape[0]
    assert np.all(np.diff(t_eval) > 0)
    assert (
        n_patches
        == K_ext_v_plate.shape[0]
        == v_plate_vec.shape[0]
        == v_init.shape[0]
        == alpha_h_vec.shape[0]
    )
    assert not np.logical_xor(delta_tau_bounded_nonuni is None, slip_taper_vec is None)

    # initialize arrays
    s_minus_upper = np.zeros((n_patches, 2))
    # tau_minus_upper = rho * np.broadcast_to(alpha_h_vec, (n_patches, 2))
    tau_minus_upper = np.arcsinh(v_init / v_plate_vec * np.exp(rho)) * alpha_h_vec
    sim_state = np.full((2 * n_patches, 2, n_eval), np.nan)
    state_plus = np.concatenate((s_minus_upper, tau_minus_upper))

    # make flat ODE function arguments
    args = (n_patches, v_plate_vec, K_int, K_ext_v_plate, rho, alpha_h_vec, mu_over_2vs)

    # integrate
    spun_up = 0
    i_slip = 0
    steps = np.sort(np.concatenate((i_eq, i_break)))
    i = 0
    if delta_tau_bounded_nonuni is not None:
        # need to get the indices of the last n_slips_obs events
        n_slips_obs = delta_tau_bounded_nonuni.shape[0]
        i_eq_nonuni = i_eq[-n_slips_obs:].tolist()
    else:
        i_eq_nonuni = []
    if verbose:
        print("0", end="", flush=True)
        ii = 1
    while i < steps.size - 1:
        # get indices
        ji, jf = steps[i], steps[i + 1]
        ti, tf = t_eval[ji], t_eval[jf]
        # call integrator
        sol = solve_ivp(
            flat_ode_rdreg,
            t_span=[ti, tf],
            y0=state_plus.ravel(),
            t_eval=t_eval[ji : jf + 1],
            method="RK45",
            args=args,
            atol=atol,
            rtol=rtol,
        )
        success = sol.success
        if success:
            sol = sol.y.reshape(2 * n_patches, 2, -1)
        else:
            raise RuntimeError("Integrator failed.")
        # save state to output array
        sim_state[:, :, ji : jf + 1] = sol
        # can already stop here if this is the last interval
        if i == steps.size - 2:
            break
        # at the end of a full cycle, check the early stopping criteria
        if (not spun_up) and (i > n_slips) and (jf in i_break):
            if verbose:
                print(f"{ii}", end="", flush=True)
                ii += 1
            old_full_state = sim_state[
                :, :, steps[i - 2 * n_slips - 1] : steps[i - n_slips]
            ]
            new_full_state = sim_state[:, :, steps[i - n_slips] : steps[i + 1]]
            old_tau_upper = old_full_state[n_patches:, :, -1]
            new_tau_upper = new_full_state[n_patches:, :, -1]
            lhs_upper = np.abs(new_tau_upper - old_tau_upper)
            rhs_upper = spinup_atol + spinup_rtol * np.fmax(
                np.abs(new_tau_upper), np.abs(old_tau_upper)
            )
            stop_now = np.all(lhs_upper <= rhs_upper)
            if stop_now:
                spun_up = jf
                # advance i to the last cycle (don't forget the general advance later)
                if verbose:
                    print(">", end="", flush=True)
                i = steps.size - n_slips - 3
        elif (not spun_up) and (jf in i_break) and verbose:
            print(f"{ii}", end="", flush=True)
            ii += 1
        elif spun_up and (jf in i_break):
            break
        # apply step change only if there is one
        if jf in i_eq:
            if verbose:
                print(".", end="", flush=True)
            state_upper = sol[:, :, -1]
            s_minus_upper = state_upper[:n_patches, :]
            tau_minus_upper = state_upper[n_patches:, :]
            s_plus_upper = s_minus_upper.copy()
            if spun_up and (jf in i_eq_nonuni):
                i_slip_nonuni = i_eq_nonuni.index(jf)
                s_plus_upper = (
                    s_minus_upper + slip_taper_vec_nonuni[i_slip_nonuni, :, :]
                )
                tau_plus_upper = (
                    tau_minus_upper + delta_tau_bounded_nonuni[i_slip_nonuni, :, :]
                )
            else:
                s_plus_upper = s_minus_upper + slip_taper_vec[i_slip, :, :]
                tau_plus_upper = tau_minus_upper + delta_tau_bounded[i_slip, :, :]
            if v_max is not None:
                v_plus_upper = (
                    v_plate_vec * np.exp(-rho) * np.sinh(tau_plus_upper / alpha_h_vec)
                )
                v_plus_upper_norm = np.linalg.norm(v_plus_upper, axis=1)
                if np.any(v_plus_upper_norm > v_max):
                    v_plus_upper *= (
                        np.clip(v_plus_upper_norm, a_min=None, a_max=v_max)
                        / v_plus_upper_norm
                    )[:, None]
                    tau_plus_upper = (
                        np.arcsinh((v_plus_upper / v_plate_vec) * np.exp(rho))
                        * alpha_h_vec
                    )
            state_plus = np.concatenate((s_plus_upper, tau_plus_upper))
            i_slip = (i_slip + 1) % n_slips
        else:
            state_plus = sol[:, :, -1]
        # advance
        i += 1

    # warn if we never spun up
    if not spun_up:
        print(f"Simulation did not spin up after {len(i_break) - 1} cycles!")

    # convert from traction to linear velocity
    sim_state[n_patches:, :, :] = (
        v_plate_vec[:, :, None]
        * np.exp(-rho)
        * np.sinh(sim_state[n_patches:, :, :] / alpha_h_vec[:, :, None])
    )

    # done
    return sim_state


def get_surface_displacements(obs_slip_ts, G):
    """
    Calculate the surface displacements given a slip timeseries and the Green's functions.

    Parameters
    ----------
    slip_ts : numpy.ndarray
        Array of shape `(n_patches, 2, n_eval)` where `n_patches` is the number
        of patches and `n_eval` is the number of timesteps.
    G : numpy.ndarray
        4D tensor of Green's functions of shape :math:`(m, 3, n, 3)`.

    Returns
    -------
    surf_disp : numpy.ndarray
        Surface displacement timeseries.
    """
    # this is equivalent to calling the single-timestep tensor product in a loop
    # over the n_eval time steps
    return np.tensordot(G[:, :, :, :2], obs_slip_ts, axes=[(3, 2), (1, 0)])


def rotation_matrix_around_vector(theta, vector):
    """
    Calculates the rotation matrix used to rotate a right-handed coordinate system
    counterclockwise around a vector when multiplied from the right.

    Parameters
    ----------
    theta : float
        Rotation angle [rad]
    vector : numpy.ndarray
        Unit rotation vector

    Returns
    -------
    R : numpy.ndarray
        Rotation matrix
    """
    assert vector.size == 3
    return (
        np.cos(theta) * np.eye(3)
        + np.sin(-theta) * np.cross(vector, -np.eye(3))
        + (1 - np.cos(theta)) * np.outer(vector, vector)
    )


def eulerpole2rotvec(euler_pole, euler_pole_covariance=None):
    r"""
    Convert an Euler pole (and optionally, its formal covariance) into a rotation
    vector and associated covariance matrix. Based on [goudarzi14]_.

    Parameters
    ----------
    euler_pole
        NumPy Array containing the longitude [rad], latitude [rad], and rotation
        rate [rad/time] of the Euler pole.
    euler_pole_covariance
        If ``rotation_covariance`` was given, the propagated uncertainty for the Euler
        Pole for all three components.

    Returns
    -------
    rotation_vector
        Rotation vector [rad/time] containing the diagonals of the :math:`3 \times 3`
        rotation matrix specifying the Euler pole in cartesian, ECEF coordinates.
    rotation_covariance
        If ``euler_pole_covariance`` was given, formal :math:`3 \times 3` covariance
        matrix [rad^2/time^2] of the rotation vector.

    See Also
    --------
    rotvec2eulerpole : Inverse function
    """
    # readability
    Ω = euler_pole[2]
    sinΩlat, cosΩlat = np.sin(euler_pole[1]), np.cos(euler_pole[1])
    sinΩlon, cosΩlon = np.sin(euler_pole[0]), np.cos(euler_pole[0])
    # rotation vector, eq. 5 (no scaling)
    ω_x = Ω * cosΩlat * cosΩlon
    ω_y = Ω * cosΩlat * sinΩlon
    ω_z = Ω * sinΩlat
    rotation_vector = np.array([ω_x, ω_y, ω_z])
    # uncertainty, eq. 6 (no scaling)
    if euler_pole_covariance is not None:
        jac = np.array(
            [
                [-Ω * cosΩlat * sinΩlon, -Ω * sinΩlat * cosΩlon, cosΩlat * cosΩlon],
                [Ω * cosΩlat * cosΩlon, -Ω * sinΩlat * sinΩlon, cosΩlat * sinΩlon],
                [0, Ω * cosΩlat, sinΩlat],
            ]
        )
        rotation_covariance = jac @ euler_pole_covariance @ jac.T
    # return
    if euler_pole_covariance is not None:
        return rotation_vector, rotation_covariance
    else:
        return rotation_vector


class Rheology(ABC):
    """
    Abstract base class for rheologies.
    """


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

    def __init__(
        self,
        v_0,
        alpha_h,
        alpha_h_mid=None,
        mid_transition=None,
        alpha_h_deep=None,
        deep_transition=None,
        boundary_width=None,
        alpha_h_boundary=None,
    ):
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
        assert not np.logical_xor(alpha_h_boundary is None, boundary_width is None)
        assert float(v_0) > 0, "RateStateSteadyLogarithmic needs to have positive v_0."
        # set number of variables
        self.n_vars = 2
        """ Number of variables to track by rheology [-] """
        # initialization
        self.v_0 = float(v_0)
        """ Reference velocity :math:`v_0` [m/s] """
        self.alpha_h = float(alpha_h)
        r""" Rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        self.alpha_h_mid = (
            float(alpha_h_mid) if alpha_h_mid is not None else self.alpha_h
        )
        r""" Middle rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        self.alpha_h_deep = (
            float(alpha_h_deep) if alpha_h_deep is not None else self.alpha_h_mid
        )
        r""" Deep rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        self.alpha_h_boundary = (
            float(alpha_h_boundary)
            if alpha_h_boundary is not None
            else self.alpha_h_deep
        )
        r""" Boundary-layer rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        self.mid_transition = None if mid_transition is None else float(mid_transition)
        """ Depth [m] for the middle transition point """
        self.deep_transition = (
            None if deep_transition is None else float(deep_transition)
        )
        """ (Upper) Depth [m] for the deep transition point """
        self.boundary_width = None if boundary_width is None else float(boundary_width)
        """ (Downdip) Width [m] of the deep transition point """

    def get_param_vectors(self, patch_centroids, outer_vertices):
        r"""
        Calculate the depth-dependent array of :math:`\alpha_h`, assuming it
        varies log-linearly with depth.
        """

        # input checks
        patch_depths = -patch_centroids[:, 2]
        assert np.all(patch_depths >= 0)

        # first part: general depth-dependent alpha_h
        # start knots list
        knots = [np.min(patch_depths)]
        vals_alpha_h = [self.alpha_h]
        # add optional mid transition
        if self.mid_transition is not None:
            knots.append(self.mid_transition)
            vals_alpha_h.append(self.alpha_h_mid)
        # add optional deep transition
        if self.deep_transition is not None:
            knots.append(self.deep_transition)
            vals_alpha_h.append(self.alpha_h_deep)
        # add final value if necessary for constant alpha_h
        if len(knots) == 1:
            knots.append(np.max(patch_depths))
            vals_alpha_h.append(self.alpha_h_deep)
        vals_alpha_h = np.array(vals_alpha_h)
        # interpolate alpha_n and alpha_eff
        alpha_h_vec = 10 ** interp1d(
            knots, np.log10(vals_alpha_h), fill_value="extrapolate"
        )(patch_depths)

        # second part: make borders to non-simulated patches gradually stronger
        if self.alpha_h_boundary is not None:
            # distances from inner patch centroids to vertices
            dists = cdist(patch_centroids, outer_vertices)
            # minimum distances
            min_dists = np.min(dists, axis=1)
            # alpha_h_vec if this distance was the only metric
            boundary_vec = (
                (self.boundary_width - min_dists)
                / self.boundary_width
                * np.log10(self.alpha_h_boundary)
            )
            # final alpha_h_vec is the element-wise maximum of the two
            alpha_h_vec = np.fmax(10**boundary_vec, alpha_h_vec)

        # done
        return alpha_h_vec


class RateStateSteadyLogarithmic2D(Rheology):
    r"""
    Implement a 2D steady-state rate-and-state rheology using the aging law (effectively
    becoming a rate-dependent rheology) with velocity in logarithmic space defined by

    :math:`f_{ss} = f_0 + (a - b) * \zeta = \tau / \sigma_E`

    where :math:`f_{ss}` is the steady-state friction, :math:`f_0` is a reference
    friction, :math:`a` and :math:`b` are the rate-and-state frictional parameters,
    :math:`\zeta = \log (v / v_0)` is the logarithmic velocity, :math:`\tau` is the shear
    stress, and :math:`\sigma_E` is the effective fault normal stress.

    The 2D spatial variation is implemented with B-Splines. Towards the boundaries
    of the simulated regions, the rheological strength is forced to a certain value
    regardless of the 2D spline-based variability.
    """

    # helper class
    class CustomBasisFunction:
        def __init__(self, full_knots_depth, full_knots_horiz, degree, ix_d, ix_h):
            self.bd = BSpline.basis_element(
                full_knots_depth[ix_d : ix_d + degree + 2], extrapolate=False
            )
            self.bh = BSpline.basis_element(
                full_knots_horiz[ix_h : ix_h + degree + 2], extrapolate=False
            )

        def __call__(self, d, h):
            return np.nan_to_num(self.bd(d)) * np.nan_to_num(self.bh(h))

    def __init__(
        self,
        v_0,
        alpha_h_mat,
        degree=1,  # knots_depth=None, knots_horiz=None,
        boundary_width=None,
        alpha_h_boundary=None,
    ):
        r"""
        Setup the rheology parameters for a given fault.

        Parameters
        ----------
        v_0 : float
            Reference velocity [m/s] used for the transformation into logarithmic space.
        alpha_h_mat : numpy.ndarray
            2D rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa],
            where :math:`a` and :math:`b` [-] are the rate-and-state frictional properties,
            and :math:`\sigma_E` [Pa] is effective fault normal stress.
            The first dimension corresponds to depth, the second to horizontal distance.
        degree : int
            Degree [-] of the B-Splines used to represent the 2D variability of `alpha_h_mat`.
        # knots_depth : list, optional
        #     If not `None`, a sorted list of knot values between 0 and 1 of fractional depths
        #     where the knots for the spline interpolation for `alpha_h_mat` are located.
        #     Defaults to uniform knot spacing.
        # knots_horiz : list, optional
        #     Same as `knots_depth` for for the horizontal direction.
        boundary_width : float
            The width of the transition zone [m] to the boundary `alpha_h` value.
        alpha_h_boundary : float
            The value of `alpha_h` [Pa] at the boundaries of the simulation.
        """

        # input check
        assert not np.logical_xor(alpha_h_boundary is None, boundary_width is None)
        assert float(v_0) > 0, "RateStateSteadyLogarithmic needs to have positive v_0."
        assert (
            isinstance(alpha_h_mat, np.ndarray) and alpha_h_mat.ndim == 2
        ), "'alpha_h_mat' needs to be a 2D NumPy array."

        # initialization
        self.n_vars = 2
        """ Number of variables to track by rheology [-] """
        self.v_0 = float(v_0)
        """ Reference velocity :math:`v_0` [m/s] """
        self.alpha_h_mat = alpha_h_mat.astype(float)
        r""" Rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """
        self.degree = int(degree)
        """ Degree of the B-Splines [-] """
        self.boundary_width = None if boundary_width is None else float(boundary_width)
        """ Width [m] of the boundary zone """
        self.alpha_h_boundary = (
            None if alpha_h_boundary is None else float(alpha_h_boundary)
        )
        r""" Boundary rate-and-state parameter :math:`(a - b) * \sigma_E` [Pa] """

        # prepare 2d spline creation
        self.num_bases_depth = self.alpha_h_mat.shape[0]
        """ Number of bases [-] in the depth direction """
        self.num_bases_horiz = self.alpha_h_mat.shape[1]
        """ Number of bases [-] in the horizontal direction """
        num_knots_depth = self.num_bases_depth - self.degree + 1
        num_knots_horiz = self.num_bases_horiz - self.degree + 1
        inner_knots_depth = (
            np.linspace(0, 1, num=num_knots_depth)
            if num_knots_depth > 1
            else np.array([])
        )
        inner_knots_horiz = (
            np.linspace(0, 1, num=num_knots_horiz)
            if num_knots_horiz > 1
            else np.array([])
        )
        full_knots_depth = np.concatenate(
            [[0] * self.degree, inner_knots_depth, [1] * self.degree]
        )
        full_knots_horiz = np.concatenate(
            [[0] * self.degree, inner_knots_horiz, [1] * self.degree]
        )

        # create 2D basis functions
        self.bases = np.array(
            [
                [
                    RateStateSteadyLogarithmic2D.CustomBasisFunction(
                        full_knots_depth, full_knots_horiz, self.degree, ix_d, ix_h
                    )
                    for ix_h in range(self.num_bases_horiz)
                ]
                for ix_d in range(self.num_bases_depth)
            ]
        )
        """ Matrix of 2D spline basis functions """

    def get_param_vectors(
        self, patch_depths, patch_horiz, patch_centroids, outer_vertices
    ):
        r"""
        Calculate the 2D-varying array of :math:`\alpha_h` and how it is made.
        """
        # first part: general 2d-varying alpha_h
        bases_eval = np.stack(
            [b(patch_depths, patch_horiz) for b in self.bases.ravel()], axis=-1
        ).reshape(patch_depths.size, self.num_bases_depth, self.num_bases_horiz)
        alpha_h_vec_raw = 10 ** np.sum(
            bases_eval * np.log10(self.alpha_h_mat[None, :, :]), axis=(1, 2)
        )
        # get flattened indices of dominant bases
        dominant_bases = np.argmax(bases_eval.reshape(alpha_h_vec_raw.size, -1), axis=1)

        # second part: make borders to non-simulated patches gradually stronger
        if self.alpha_h_boundary is None:
            alpha_h_vec = alpha_h_vec_raw
        else:
            # distances from inner patch centroids to vertices
            dists = cdist(patch_centroids, outer_vertices)
            # minimum distances
            min_dists = np.min(dists, axis=1)
            # alpha_h_vec if this distance was the only metric
            boundary_vec = 10 ** (
                (self.boundary_width - min_dists)
                / self.boundary_width
                * np.log10(self.alpha_h_boundary)
            )
            # final alpha_h_vec is the element-wise maximum of the two
            alpha_h_vec = np.fmax(boundary_vec, alpha_h_vec_raw)
            # set to NaN where the boundary condition took over
            bc_strongest = np.argmax(
                np.vstack([alpha_h_vec_raw, boundary_vec]), axis=0
            ).astype(bool)
            dominant_bases[bc_strongest] = -1

        # done
        return alpha_h_vec, dominant_bases

    def get_valid_domain(self, patch_depths, patch_horiz, ix_estim_rows, ix_estim_cols):
        """
        Calculate the domain where the given row and column indices are unaffected
        by the other indices.
        """
        # start again with bases evaluation
        bases_eval = np.stack(
            [b(patch_depths, patch_horiz) for b in self.bases.ravel()], axis=-1
        ).reshape(patch_depths.size, self.num_bases_depth, self.num_bases_horiz)
        # find all non-estimated rows and columns
        ix_const_rows = [
            i for i in range(self.num_bases_depth) if i not in ix_estim_rows
        ]
        ix_const_cols = [
            i for i in range(self.num_bases_horiz) if i not in ix_estim_cols
        ]
        # find all evaluation indices that have non-zero effects from ix_const_*
        ix_const_mat = ~np.logical_or(
            np.any(
                bases_eval[
                    np.ix_(
                        list(range(patch_depths.size)),
                        ix_const_rows,
                        list(range(self.num_bases_horiz)),
                    )
                ]
                != 0,
                axis=(1, 2),
            ),
            np.any(
                bases_eval[
                    np.ix_(
                        list(range(patch_depths.size)),
                        list(range(self.num_bases_depth)),
                        ix_const_cols,
                    )
                ]
                != 0,
                axis=(1, 2),
            ),
        )
        # done
        return ix_const_mat


class Fault3D:
    """
    Base class for the subduction fault mesh.
    """

    # shorthands for specifying farfield motion type
    FARFIELD_MOTION_UNIFORM = 1
    FARFIELD_MOTION_ROTATION = 2
    FARFIELD_MOTION_EULER = 3
    FARFIELD_MOTION_PROVIDED = 4

    def __init__(
        self,
        inner_mesh_files,
        asperity_mesh_files,
        lower_mesh_files,
        downdip_direction,
        mu,
        nu,
        v_s,
        farfield_motion,
        farfield_motion_type,
        max_simulated_distance,
        max_simulated_depth,
        orient_along=0,
        offset_lower=0,
        utmzone=None,
        lon0=None,
        lat0=None,
        K_inner_asperities=None,
        K_inner_inner=None,
        cross_sections=None,
    ):
        """
        Define the fault mesh of the subduction zone fault system, based on the
        Elastic Subducting Plate Model (ESPM) of [kanda2010]_.

        The inner mesh consists of the asperities as well as the surrounding
        creeping patches, formulated as triangular dislocation elements.

        The outer mesh consists of the boundary patches that are necessary
        to recover longterm surface displacement due to continuous fault slip.

        Parameters
        ----------
        inner_mesh_files : list
            Path to `.tet` mesh file defining the inner mesh on the upper plate
            interface.
        asperity_mesh_files : list
            List of paths to `.tet` mesh files defining the upper plate interface's
            asperities, assumed to be within the inner mesh.
        lower_mesh_files : list
            List of paths to `.tet` mesh files defining the lower plate interface.
        downdip_direction : str
            Direction into which the inner mesh should be extended using the
            normal vectors. All other directions will be extended on constant
            boundary depth. Two-character code of sign `+-` and axes `xy`.
        mu : float
            Shear modulus [Pa]
        nu : float
            Poisson's ratio [-]
        v_s : float
            Shear wave velocity [m/s] in the fault zone.
        farfield_motion : numpy.ndarray
            Either a 1D array containing the uniform plate motion [m/s] in ENU for all patches,
            or an ECEF rotation vector [°/Ma], a Longitude/Latitude/Rate Euler Pole [°],
            or a 2D array containing the precomputed ENU plate velocity [m/s] at each patch.
        farfield_motion_type : int
            Describes the type of `farfield_motion` between `1` (uniform), `2`
            (ECEF rotation), `3` (Euler Pole), or `4` (precomputed).
        max_simulated_distance : float
            Maximum horizontal distance [m] from asperities until which to simulate patches
        max_simulated_depth : float
            Maximum depth [m] (positive) until which to simulate patches
        orient_along : float
            Angle [° counter-clockwise from east] of the subduction fault strike.
        offset_lower : float
            Additional offset the lower part of the mesh [m] (positive means upwards shift)
        utmzone : int, optional
            If `farfield_motion_type` is an ECEF rotation or Euler Pole, it is necessary to
            know in which coordinate system the mesh files are.
        lon0 : float, optional
            If there is a custom longitude offset [°] used in calculating the mesh file
            UTM coordinates, provide it here.
        lat0 : float, optional
            If there is a custom latitude offset [°] used in calculating the mesh file
            UTM coordinates, provide it here.
        K_inner_asperities : numpy.ndarray, optional
            Precomputed stress kernel from the asperities to the inner mesh [Pa/m]
        K_inner_inner : numpy.ndarray, optional
            Precomputed stress kernel within the inner mesh [Pa/m]
        cross_sections : list, optional
            Precomputed valid cross sections
        """

        # input check
        assert isinstance(inner_mesh_files, list)
        assert isinstance(asperity_mesh_files, list)
        assert isinstance(lower_mesh_files, list)
        assert isinstance(downdip_direction, str)
        assert 1 <= farfield_motion_type <= 4
        assert isinstance(farfield_motion_type, int)
        if farfield_motion_type == 4:
            assert (
                isinstance(farfield_motion, np.ndarray)
                and (farfield_motion.ndim == 2)
                and (farfield_motion.shape[1] == 3)
            )
        else:
            assert (
                isinstance(farfield_motion, np.ndarray)
                and (farfield_motion.ndim == 1)
                and (farfield_motion.size == 3)
            )
        if farfield_motion_type in (2, 3):
            assert isinstance(utmzone, int), (
                f"For 'farfield_motion_type={farfield_motion_type}', "
                "an integer 'utmzone' needs to be provided."
            )
        use_normals = [False, False, False, False]
        if downdip_direction[0] == "+":
            if downdip_direction[1] == "x":
                use_normals[0] = True
            elif downdip_direction[1] == "y":
                use_normals[1] = True
        elif downdip_direction[0] == "-":
            if downdip_direction[1] == "x":
                use_normals[2] = True
            elif downdip_direction[1] == "y":
                use_normals[3] = True
        if not any(use_normals):
            raise ValueError(f"Can't parse 'use_normals' from {downdip_direction}.")

        # material properties
        self.mu = float(mu)
        """ Shear modulus [Pa] """
        self.nu = float(nu)
        """ Poisson's ratio [-] """
        self.v_s = float(v_s)
        """ Shear wave velocity [m/s] in the fault zone """
        self.E = 2 * self.mu * (1 + self.nu)
        """ Young's modulus [Pa] of the fault zone """
        self.mu_over_2vs = self.E / (2 * (1 + self.nu) * 2 * self.v_s)
        """ Radiation damping term [Pa * s/m] """
        self.farfield_motion_type = farfield_motion_type
        """ Type of farfield motion used to calculate plate velocity """
        self.utmzone = int(utmzone) if utmzone is not None else None
        """ UTM zone number of fault coordinates """

        # simulation boundaries
        assert max_simulated_distance > 0
        self.max_simulated_distance = float(max_simulated_distance)
        """ Maximum horizontal distance [m] from asperities until which to simulate patches """
        assert max_simulated_depth > 0
        self.max_simulated_depth = float(max_simulated_depth)
        """ Maximum depth [m] (positive) until which to simulate patches """
        self.orient_along = float(orient_along)
        """ Main orientation of the subduction zone [°] """
        self.offset_lower = float(offset_lower)
        """ Offset applied to the lower mesh files [m] """

        # create asperities mesh
        vertices_asperities, triangles_asperities, match_sources = combine_tet(
            *asperity_mesh_files, return_sources=True
        )
        # correct custom UTM by applying average shift
        if (self.utmzone is not None) and ((lon0 is not None) or (lat0 is not None)):
            offset_en = (
                correct_custom_utm(vertices_asperities[:, :2], self.utmzone, lon0, lat0)
                - vertices_asperities[:, :2]
            ).mean(axis=0)
            vertices_asperities[:, :2] += offset_en
        assert (
            vertices_asperities[:, 2].max() <= 0
        ), "Found up-coordinates larger than 0 in the asperities mesh."
        vertices_asperities, triangles_asperities = force_normals(
            vertices_asperities, triangles_asperities, normals_up=False
        )
        self.asperities_vertices = vertices_asperities
        """ Vertex coordinates for the asperities mesh [m] """
        self.asperities_triangles = triangles_asperities
        """ Triangle definitions for the asperities mesh [-] """
        self.asperities_patches = self.asperities_vertices[self.asperities_triangles]
        """ Asperities patches """
        self.asperities_polygons = [
            shapely.Polygon(arr) for arr in self.asperities_patches
        ]
        """ List of all asperity triangles as Shapely Polygons """
        self.asperities_centroids = np.mean(self.asperities_patches, axis=1)
        """ Centroids of asperities patches [m] """
        self.asperities_num_vertices = self.asperities_vertices.shape[0]
        """ Number of asperities vertices [-] """
        self.asperities_num_patches = self.asperities_triangles.shape[0]
        """ Number of asperities patches/triangles [-] """
        self.asperity_indices = match_sources
        """ List of indices matching each triangle to an asperity """
        self.num_asperities = len(self.asperity_indices)
        """ Number of distinct asperities """

        # create asperities rotation matrices
        self.R_asperities_efcs_to_tdcs = compute_efcs_to_tdcs_rotations(
            self.asperities_patches
        )
        """ Rotation matrices from EFCS to TDCS for the asperities patches """

        # create inner mesh
        vertices_inner, triangles_inner = combine_tet(*inner_mesh_files)
        if (self.utmzone is not None) and ((lon0 is not None) or (lat0 is not None)):
            vertices_inner[:, :2] += offset_en
        assert (
            vertices_inner[:, 2].max() <= 0
        ), "Found up-coordinates larger than 0 in the inner mesh."
        vertices_inner, triangles_inner = force_normals(
            vertices_inner, triangles_inner, normals_up=False
        )
        # extend upper plate interface
        extend_directions = ~np.roll(use_normals, 2)
        R_inner_efcs_to_tdcs = compute_efcs_to_tdcs_rotations(
            vertices_inner[triangles_inner]
        )
        vertices_combined, triangles_combined = extend_mesh(
            vertices_inner,
            triangles_inner,
            use_normals=use_normals,
            directions=extend_directions,
            normals_in=R_inner_efcs_to_tdcs[:, 2, :],
        )
        vertices_inf = vertices_combined[vertices_inner.shape[0] :, :].copy()
        vertices_combined, triangles_combined = force_normals(
            vertices_combined, triangles_combined, normals_up=False
        )
        vertices_simulated, vertices_imposed, triangles_simulated, triangles_imposed = (
            split_mesh(
                vertices_combined,
                triangles_combined,
                self.asperities_patches,
                max_distance=self.max_simulated_distance,
                max_depth=self.max_simulated_depth,
            )
        )
        ix_v_inf = np.concatenate(
            [
                np.flatnonzero(np.all(vertices_imposed == v, axis=1))
                for v in vertices_inf.tolist()
            ]
        )
        assert ix_v_inf.size == vertices_inf.shape[0]
        self.ix_tri_inf = np.any(np.isin(triangles_imposed, ix_v_inf), axis=1)
        """ Indices of infinity triangles """

        # save inner mesh
        self.inner_vertices = vertices_simulated
        """ Vertex coordinates for the inner mesh [m] """
        self.inner_triangles = triangles_simulated
        """ Triangle definitions for the inner mesh [-] """
        self.inner_patches = self.inner_vertices[self.inner_triangles]
        """ Inner patches """
        self.inner_centroids = np.mean(self.inner_patches, axis=1)
        """ Centroids of inner patches [m] """
        self.inner_num_vertices = self.inner_vertices.shape[0]
        """ Number of inner vertices [-] """
        self.inner_num_patches = self.inner_triangles.shape[0]
        """ Number of inner patches/triangles [-] """

        # create inner rotation matrices
        self.R_inner_efcs_to_tdcs = compute_efcs_to_tdcs_rotations(self.inner_patches)
        """ Rotation matrices from EFCS to TDCS for the inner patches """

        # create a joint inner and asperities mesh for 2D projection purposes
        slab_vertices = np.concatenate(
            [self.inner_vertices, self.asperities_vertices], axis=0
        )
        slab_triangles = np.concatenate(
            [
                self.inner_triangles,
                self.asperities_triangles + self.inner_vertices.shape[0],
            ],
            axis=0,
        )
        slab_vertices, slab_triangles = remove_duplicate_vertices(
            slab_vertices, slab_triangles
        )
        self.slab_vertices = slab_vertices
        """ Vertex coordinates for the slab mesh (inner + asperities) [m] """
        self.slab_triangles = slab_triangles
        """ Triangle definitions for the slab mesh (inner + asperities) [-] """

        # save outer mesh
        self.outer_vertices = vertices_imposed
        """ Vertex coordinates for the outer upper mesh [m] """
        self.outer_vertices_noninf = vertices_imposed
        """ Vertex coordinates for the outer upper mesh [m] """
        self.outer_triangles = triangles_imposed
        """ Triangle definitions for the outer upper mesh [-] """
        self.outer_triangles_noninf = triangles_imposed[~self.ix_tri_inf, :]
        """ Triangle definitions for the outer upper mesh [-], excluding extended patches """
        self.outer_patches = self.outer_vertices[self.outer_triangles]
        """ Outer upper patches """
        self.outer_centroids = np.mean(self.outer_patches, axis=1)
        """ Centroids of outer upper patches [m] """
        self.outer_num_vertices = self.outer_vertices.shape[0]
        """ Number of outer upper vertices [-] """
        self.outer_num_patches = self.outer_triangles.shape[0]
        """ Number of outer upper patches/triangles [-] """

        # create outer upper rotation matrices
        self.R_outer_efcs_to_tdcs = compute_efcs_to_tdcs_rotations(self.outer_patches)
        """ Rotation matrices from EFCS to TDCS for the outer patches """

        # create lower plate interface mesh
        vertices_lower, triangles_lower = combine_tet(*lower_mesh_files)
        if (self.utmzone is not None) and ((lon0 is not None) or (lat0 is not None)):
            vertices_lower[:, :2] += offset_en
        vertices_lower[:, 2] += self.offset_lower
        assert (
            vertices_lower[:, 2].max() <= 0
        ), "Found up-coordinates larger than 0 in the lower mesh."
        # extend in all directions
        R_lower_efcs_to_tdcs = compute_efcs_to_tdcs_rotations(
            vertices_lower[triangles_lower]
        )
        vertices_lower_extended, triangles_lower_extended = extend_mesh(
            vertices_lower,
            triangles_lower,
            use_normals=use_normals,
            normals_in=R_lower_efcs_to_tdcs[:, 2, :],
        )
        self.num_lower_extended = (
            triangles_lower_extended.shape[0] - triangles_lower.shape[0]
        )
        """ Number of triangles that were added to the lower plate interface """
        vertices_lower_extended, triangles_lower_extended = force_normals(
            vertices_lower_extended, triangles_lower_extended, normals_up=True
        )
        self.lower_vertices = vertices_lower_extended
        """ Vertex coordinates for the lower mesh [m] """
        self.lower_triangles = triangles_lower_extended
        """ Triangle definitions for the lower mesh [-] """
        self.lower_patches = self.lower_vertices[self.lower_triangles]
        """ Lower patches """
        self.lower_centroids = np.mean(self.lower_patches, axis=1)
        """ Centroids of lower patches [m] """
        self.lower_num_vertices = self.lower_vertices.shape[0]
        """ Number of lower vertices [-] """
        self.lower_num_patches = self.lower_triangles.shape[0]
        """ Number of lower patches/triangles [-] """

        # create lower rotation matrices
        self.R_lower_efcs_to_tdcs = compute_efcs_to_tdcs_rotations(self.lower_patches)
        """ Rotation matrices from EFCS to TDCS for the lower patches """

        # create a joint R tensor
        self.R_joint_efcs_to_tdcs = np.concatenate(
            [
                self.R_inner_efcs_to_tdcs,
                self.R_asperities_efcs_to_tdcs,
                self.R_outer_efcs_to_tdcs,
                self.R_lower_efcs_to_tdcs,
            ],
            axis=0,
        )
        """ Rotation matrices from EFCS to TDCS for all patches """

        # create a joint centroids array
        self.joint_centroids = np.concatenate(
            [
                self.inner_centroids,
                self.asperities_centroids,
                self.outer_centroids,
                self.lower_centroids,
            ],
            axis=0,
        )
        """ Centroids of all patches [m] """

        # create a joint patches array
        self.joint_patches = np.concatenate(
            [
                self.inner_patches,
                self.asperities_patches,
                self.outer_patches,
                self.lower_patches,
            ],
            axis=0,
        )
        """ All patches [m] """

        # create slice objects for all subsets in the joint arrays
        self.s_inner = np.s_[0 : self.inner_num_patches]
        """ Slice object to recover the inner patches from all patches """
        self.s_asperities = np.s_[
            self.inner_num_patches : self.inner_num_patches
            + self.asperities_num_patches
        ]
        """ Slice object to recover the asperities' patches from all patches """
        self.s_outer = np.s_[
            self.inner_num_patches
            + self.asperities_num_patches : self.inner_num_patches
            + self.asperities_num_patches
            + self.outer_num_patches
        ]
        """ Slice object to recover the outer patches from all patches """
        self.s_lower = np.s_[
            self.inner_num_patches
            + self.asperities_num_patches
            + self.outer_num_patches : self.inner_num_patches
            + self.asperities_num_patches
            + self.outer_num_patches
            + self.lower_num_patches
        ]
        """ Slice object to recover the lower patches from all patches """
        self.joint_num_patches = (
            self.inner_num_patches
            + self.asperities_num_patches
            + self.outer_num_patches
            + self.lower_num_patches
        )
        """ Total number of patches """

        # calculate farfield motion unit vector
        if farfield_motion_type in (
            self.FARFIELD_MOTION_UNIFORM,
            self.FARFIELD_MOTION_PROVIDED,
        ):
            v_plate_vec = np.ascontiguousarray(
                np.broadcast_to(farfield_motion, (self.joint_num_patches, 3))
            )
        elif farfield_motion_type in (
            self.FARFIELD_MOTION_ROTATION,
            self.FARFIELD_MOTION_EULER,
        ):
            # convert rotation format if necessary
            if farfield_motion_type == self.FARFIELD_MOTION_EULER:
                farfield_motion = np.rad2deg(
                    eulerpole2rotvec(np.deg2rad(farfield_motion))
                )
            # define coordinate systems
            crs_lla = ccrs.Geodetic()
            crs_xyz = ccrs.Geocentric()
            crs_utm = ccrs.UTM(zone=self.utmzone)
            # get locations of centroids in lon/lat and ECEF cartesian
            centroids_lla = crs_lla.transform_points(
                crs_utm, self.joint_centroids[:, 0], self.joint_centroids[:, 1]
            )
            centroids_xyz = crs_xyz.transform_points(
                crs_utm, self.joint_centroids[:, 0], self.joint_centroids[:, 1]
            )
            # calculate rotation from rotation vector
            farfield_motion_rad_s = np.deg2rad(farfield_motion) / 1e6 / SEC_PER_YEAR
            centroids_vel_xyz = np.cross(farfield_motion_rad_s, centroids_xyz)
            # project into local ENU
            v_plate_vec = np.stack(
                [
                    (R_ecef2enu(lo, la) @ centroids_vel_xyz[i, :])
                    for i, (lo, la) in enumerate(
                        zip(centroids_lla[:, 0], centroids_lla[:, 1])
                    )
                ]
            )
            # the infinitely extending patches are probably not covered by this
            # approximation and will yield NaNs, so we assign the same plate
            # velocity to them as the nearest non-infinite patch
            ix_inf = (
                np.flatnonzero(self.ix_tri_inf)
                + self.inner_num_patches
                + self.asperities_num_patches
            ).tolist() + (
                np.arange(self.lower_num_patches)[-self.num_lower_extended :]
                + self.inner_num_patches
                + self.asperities_num_patches
                + self.outer_num_patches
            ).tolist()
            ix_notinf = ~np.isin(np.arange(self.joint_num_patches), ix_inf)
            # loop over infinite patch indices
            for i in ix_inf:
                # find closest non-infinite patch
                j = np.argmin(
                    np.linalg.norm(
                        self.joint_centroids[ix_notinf, :]
                        - self.joint_centroids[i, :][None, :],
                        axis=1,
                    )
                )
                v_plate_vec[i, :] = v_plate_vec[ix_notinf, :][j, :]

        self.v_plate_vec = v_plate_vec
        """ Nominal far-field plate velocity at all patches in the dimensions of the rheology """

        # project v_plate_vec onto each patch
        self.e_plate_tdcs = Fault3D.vectors_forward_rotation(
            self.v_plate_vec / np.linalg.norm(self.v_plate_vec, axis=1, keepdims=True),
            self.R_joint_efcs_to_tdcs,
        )
        """ Unit plate velocity direction in the TDCS [m] """
        self.e_plate_tdcs_proj = Fault3D.redistribute_normal(self.e_plate_tdcs)
        """ Projected unit plate velocity direction in TDCS [m] """

        # create DDCS
        # angle between plate velocity and TDCS strike (already on same plane)
        alpha = np.arctan2(self.e_plate_tdcs_proj[:, 1], self.e_plate_tdcs_proj[:, 0])
        # inner
        temp_transform_R = np.array(
            [
                rotation_matrix_around_vector(
                    al - np.pi / 4, self.R_inner_efcs_to_tdcs[i, 2, :]
                )
                for i, al in enumerate(alpha[self.s_inner])
            ]
        )
        self.R_inner_efcs_to_ddcs = np.einsum(
            "ijk,ikl->ijl", self.R_inner_efcs_to_tdcs, temp_transform_R
        )
        """ Rotation matrices from EFCS to DDCS for the inner patches """
        self.R_inner_tdcs_to_ddcs = np.einsum(
            "ijk,ilk->ijl", self.R_inner_efcs_to_ddcs, self.R_inner_efcs_to_tdcs
        )
        """ Rotation matrices from TDCS to DDCS for the inner patches """
        # asperities
        temp_transform_R = np.array(
            [
                rotation_matrix_around_vector(
                    al - np.pi / 4, self.R_asperities_efcs_to_tdcs[i, 2, :]
                )
                for i, al in enumerate(alpha[self.s_asperities])
            ]
        )
        self.R_asperities_efcs_to_ddcs = np.einsum(
            "ijk,ikl->ijl", self.R_asperities_efcs_to_tdcs, temp_transform_R
        )
        """ Rotation matrices from EFCS to DDCS for the asperities patches """
        self.R_asperities_tdcs_to_ddcs = np.einsum(
            "ijk,ilk->ijl",
            self.R_asperities_efcs_to_ddcs,
            self.R_asperities_efcs_to_tdcs,
        )
        """ Rotation matrices from TDCS to DDCS for the asperities patches """
        # outer
        temp_transform_R = np.array(
            [
                rotation_matrix_around_vector(
                    al - np.pi / 4, self.R_outer_efcs_to_tdcs[i, 2, :]
                )
                for i, al in enumerate(alpha[self.s_outer])
            ]
        )
        self.R_outer_efcs_to_ddcs = np.einsum(
            "ijk,ikl->ijl", self.R_outer_efcs_to_tdcs, temp_transform_R
        )
        """ Rotation matrices from EFCS to DDCS for the outer patches """
        self.R_outer_tdcs_to_ddcs = np.einsum(
            "ijk,ilk->ijl", self.R_outer_efcs_to_ddcs, self.R_outer_efcs_to_tdcs
        )
        """ Rotation matrices from TDCS to DDCS for the outer patches """
        # lower
        temp_transform_R = np.array(
            [
                rotation_matrix_around_vector(
                    al - np.pi / 4, self.R_lower_efcs_to_tdcs[i, 2, :]
                )
                for i, al in enumerate(alpha[self.s_lower])
            ]
        )
        self.R_lower_efcs_to_ddcs = np.einsum(
            "ijk,ikl->ijl", self.R_lower_efcs_to_tdcs, temp_transform_R
        )
        """ Rotation matrices from EFCS to DDCS for the lower patches """
        self.R_lower_tdcs_to_ddcs = np.einsum(
            "ijk,ilk->ijl", self.R_lower_efcs_to_ddcs, self.R_lower_efcs_to_tdcs
        )
        """ Rotation matrices from TDCS to DDCS for the lower patches """

        # create more joint R tensors
        self.R_joint_efcs_to_ddcs = np.concatenate(
            [
                self.R_inner_efcs_to_ddcs,
                self.R_asperities_efcs_to_ddcs,
                self.R_outer_efcs_to_ddcs,
                self.R_lower_efcs_to_ddcs,
            ],
            axis=0,
        )
        """ Rotation matrices from EFCS to DDCS for all patches """
        self.R_joint_tdcs_to_ddcs = np.concatenate(
            [
                self.R_inner_tdcs_to_ddcs,
                self.R_asperities_tdcs_to_ddcs,
                self.R_outer_tdcs_to_ddcs,
                self.R_lower_tdcs_to_ddcs,
            ],
            axis=0,
        )
        """ Rotation matrices from TDCS to DDCS for all patches """

        # project v_plate_vec onto each patch again, this time DDCS
        self.e_plate_ddcs_proj = Fault3D.vectors_forward_rotation(
            self.e_plate_tdcs_proj, self.R_joint_tdcs_to_ddcs[:, :2, :2]
        )
        """ Projected unit plate velocity direction in DDCS [m] """
        e_plate_tdcs_proj_3d = np.concatenate(
            [self.e_plate_tdcs_proj, np.zeros((self.e_plate_tdcs_proj.shape[0], 1))],
            axis=1,
        )
        e_plate_efcs_proj_3d = Fault3D.vectors_backward_rotation(
            e_plate_tdcs_proj_3d, self.R_joint_efcs_to_tdcs
        )
        assert np.allclose(
            np.concatenate(
                [
                    self.e_plate_ddcs_proj,
                    np.zeros((self.e_plate_ddcs_proj.shape[0], 1)),
                ],
                axis=1,
            ),
            Fault3D.vectors_forward_rotation(
                e_plate_efcs_proj_3d, self.R_joint_efcs_to_ddcs
            ),
        )

        # compute the relevant stress kernels in DDCS, if not already provided
        if K_inner_asperities is None:
            print("Calculating asperity stress kernel... ", flush=True, end="")
            tic = perf_counter()
            K_inner_asperities = self.get_stress_kernel(
                self.asperities_patches,
                self.inner_centroids,
                self.R_inner_efcs_to_ddcs,
                self.mu,
                self.nu,
                rot_mats_tdcs=self.R_asperities_tdcs_to_ddcs,
            )
            toc = perf_counter()
            print(f"done ({toc - tic:.2f} s)", flush=True)
        else:
            assert isinstance(
                K_inner_asperities, np.ndarray
            ), f"Expected 'K_inner_asperities' NumPy array, got {type(K_inner_asperities)}."
            expected_shape = (self.inner_num_patches, 3, self.asperities_num_patches, 3)
            assert K_inner_asperities.shape == expected_shape, (
                f"Expected 'K_inner_asperities' shape of {expected_shape}, "
                f"got {K_inner_asperities.shape}."
            )
            print(f"Accepted input 'K_inner_asperities' of shape {expected_shape}.")
        self.K_inner_asperities = K_inner_asperities
        """ Stress kernel from the asperities to the inner mesh [Pa/m] """
        if K_inner_inner is None:
            print("Calculating internal stress kernel... ", flush=True, end="")
            tic = perf_counter()
            K_inner_inner = self.get_stress_kernel(
                self.inner_patches,
                self.inner_centroids,
                self.R_inner_efcs_to_ddcs,
                self.mu,
                self.nu,
                rot_mats_tdcs=self.R_inner_tdcs_to_ddcs,
            )
            toc = perf_counter()
            print(f"done ({toc - tic:.2f} s)", flush=True)
        else:
            assert isinstance(
                K_inner_inner, np.ndarray
            ), f"Expected 'K_inner_inner' NumPy array, got {type(K_inner_inner)}."
            expected_shape = (self.inner_num_patches, 3, self.inner_num_patches, 3)
            assert K_inner_inner.shape == expected_shape, (
                f"Expected 'K_inner_inner' shape of {expected_shape}, "
                f"got {K_inner_inner.shape}."
            )
            print(f"Accepted input 'K_inner_inner' of shape {expected_shape}.")
        self.K_inner_inner = K_inner_inner
        """ Stress kernel within the inner mesh [Pa/m] """

        # calculate horizontal cross sections
        if cross_sections is None:
            print("Calculating mesh cross sections... ", flush=True, end="")
            tic = perf_counter()
            # format mesh
            trimesh = meshcut.TriangleMesh(self.slab_vertices, self.slab_triangles)
            # get all cross sections
            all_cross_sections = [
                meshcut.cross_section_mesh(
                    trimesh,
                    meshcut.Plane(self.inner_centroids[i, :], [0, 0, 1]),
                    dist_tol=1,
                )
                for i in range(self.inner_num_patches)
            ]
            # eliminate those that have multiple segments
            cross_sections = [
                cs[0] if len(cs) == 1 else None for cs in all_cross_sections
            ]
            # eliminate those that are significantly shorter than the median, so probably wrong
            cs_lengths = np.array(
                [
                    shapely.LineString(cs).length if cs is not None else np.nan
                    for cs in cross_sections
                ]
            )
            cs_length_cutoff = np.nanmedian(cs_lengths) * 0.8
            for i in np.flatnonzero(cs_lengths < cs_length_cutoff):
                cross_sections[i] = None
            toc = perf_counter()
            print(f"done ({toc - tic:.2f} s)", flush=True)
        else:
            assert isinstance(cross_sections, list) and all(
                [
                    (
                        isinstance(cs, np.ndarray)
                        and (cs.ndim == 2)
                        and (cs.shape[1] == 3)
                    )
                    or (cs is None)
                    for cs in cross_sections
                ]
            ), "Expected a list of Nones or well-shaped NumPy arrays as cross sections."
            print("Accepted input 'cross_sections'.")
        self.cross_sections = cross_sections
        """ Horizontal cross sections through the inner centroid along the slab """

        # calculate the spline coefficients onto which to interpolate rheological values
        # convert the ENU coordinates of patch_centroids into a vertical (depth) and horizontal
        # (along-strike, constant depth) coordinate system, and normalize to (0, 1)
        # vertical (increasing towards greater depth)
        assert np.all(self.inner_centroids[:, 2] <= 0)
        patch_depths = (self.inner_centroids[:, 2] - self.slab_vertices[:, 2].max()) / (
            self.slab_vertices[:, 2].min() - self.slab_vertices[:, 2].max()
        )
        # horizontal
        patch_horiz = np.zeros_like(patch_depths)
        target_dir = np.array(
            [
                np.cos(np.deg2rad(self.orient_along)),
                np.sin(np.deg2rad(self.orient_along)),
            ]
        )
        successful_lines = [None for _ in range(patch_horiz.size)]
        # get all horizontal cut lines
        for i, cs in enumerate(self.cross_sections):
            # use the supplied cross-sections
            if cs is not None:
                # we can use Shapely for the distance along the line since the z-component does
                # not vary because of how we slice
                ls = shapely.LineString(cs)
                assert ls.is_simple and ls.is_valid, "LineString not valid or simple"
                # make sure the line points in the right direction
                ls_dir = np.array(
                    [
                        ls.coords[-1][1] - ls.coords[0][1],
                        ls.coords[-1][0] - ls.coords[0][0],
                    ]
                )
                dir_diff = np.arccos(
                    np.dot(target_dir, ls_dir) / np.linalg.norm(ls_dir)
                )
                if dir_diff > np.pi / 2:
                    ls = ls.reverse()
                # save for later use
                successful_lines[i] = ls
        # calculate all horizontal distances
        for i, test_ls in enumerate(successful_lines):
            # find closest existing line if the current one didn't work
            ls = test_ls
            if ls is None:
                for test_ix in np.argsort(np.abs(patch_depths[i] - patch_depths)):
                    if successful_lines[test_ix] is not None:
                        ls = successful_lines[test_ix]
                        break
            assert ls is not None
            # now calculate the normalized distance along the cross section until the centroid
            patch_horiz[i] = ls.line_locate_point(
                shapely.Point(self.inner_centroids[i, :]), normalized=True
            )
        patch_horiz[patch_horiz == 1] -= 1e-10
        # save
        self.inner_patch_2d_coords = np.stack([patch_depths, patch_horiz], axis=1)
        """
        2D normalized coordinates of the inner patches
        along horizontal & depth directions [-]
        """

        # prepare quantities relating to slip tapering
        self.dists_inner_asperities = np.array(
            [
                [shapely.distance(p, shapely.Point(c)) for c in self.inner_centroids]
                for p in self.asperities_polygons
            ]
        )
        """
        Distances [m] between asperity triangles and inner centroids,
        shape (num_asperities, num_inner)
        """
        # for each asperity, get the indices of the triangle closest to each inner mesh centroid
        self.i_closest_asp_tri = np.array(
            [
                aix[np.argmin(self.dists_inner_asperities[aix, :], axis=0)]
                for aix in self.asperity_indices
            ]
        )
        """
        Indices of closest asperity triangle for each asperity and each inner patch,
        shape (num_asperities, num_inner)
        """

    @classmethod
    def from_cfg_and_files(
        cls,
        fault_dict,
        K_inner_asperities_file=None,
        K_inner_inner_file=None,
        cross_sections_file=None,
        write_new_files=True,
        verbose=False,
    ):
        """
        Create a `Fault3D` object from a configuration dictionary and precomputed arrays
        stored to files using `write_big_files`.

        Parameters
        ----------
        fault_dict : dict
            Dictionary of keyword arguments passed on to the initialization; from
            `SubductionSimulation3D.read_config_file`
        K_inner_asperities_file : str, optional
            File name for `K_inner_asperities`
        K_inner_inner_file : str, optional
            File name for `K_inner_inner`
        cross_sections_file : str, optional
            File name for `cross_sections`
        write_new_files : bool, optional
            Whether to save to files the kernels and cross sections if they were missing
        verbose : bool, optional
            Print what's happening

        Returns
        -------
        fault : Fault3D
            The new `Fault3D` object
        """
        # check for K_inner_asperities
        if K_inner_asperities_file is not None:
            try:
                K_inner_asperities = np.load(K_inner_asperities_file)
            except FileNotFoundError:
                K_inner_asperities = None
                if verbose:
                    print(
                        f"Couldn't find '{K_inner_asperities_file}', "
                        "need to recompute K_inner_asperities"
                    )
            else:
                if verbose:
                    print(f"Loaded K_inner_asperities from '{K_inner_asperities_file}'")
            if "K_inner_asperities" in fault_dict:
                del fault_dict["K_inner_asperities"]
        else:
            K_inner_asperities = fault_dict.pop("K_inner_asperities", None)
        # check for K_inner_inner
        if K_inner_inner_file is not None:
            try:
                K_inner_inner = np.load(K_inner_inner_file)
            except FileNotFoundError:
                K_inner_inner = None
                if verbose:
                    print(
                        f"Couldn't find '{K_inner_inner_file}', "
                        "need to recompute K_inner_inner"
                    )
            else:
                if verbose:
                    print(f"Loaded K_inner_inner from '{K_inner_inner_file}'")
            if "K_inner_inner" in fault_dict:
                del fault_dict["K_inner_inner"]
        else:
            K_inner_inner = fault_dict.pop("K_inner_inner", None)
        # check for cross_sections
        if cross_sections_file is not None:
            try:
                with open(cross_sections_file, "rb") as f:
                    cross_sections = pickle.load(f)
            except FileNotFoundError:
                cross_sections = None
                if verbose:
                    print(
                        f"Couldn't find '{cross_sections_file}', "
                        "need to recompute cross_sections"
                    )
            else:
                if verbose:
                    print(f"Loaded cross_sections from '{cross_sections_file}'")
        else:
            cross_sections = None
        # create new object
        fault = cls(
            **fault_dict,
            K_inner_asperities=K_inner_asperities,
            K_inner_inner=K_inner_inner,
            cross_sections=cross_sections,
        )
        # save new files if desired
        if write_new_files:
            fault.write_big_files(
                K_inner_asperities_file=(
                    K_inner_asperities_file
                    if (K_inner_asperities_file is not None)
                    and (K_inner_asperities is None)
                    else None
                ),
                K_inner_inner_file=(
                    K_inner_inner_file
                    if (K_inner_inner_file is not None) and (K_inner_inner is None)
                    else None
                ),
                cross_sections_file=(
                    cross_sections_file
                    if (cross_sections_file is not None) and (cross_sections is None)
                    else None
                ),
                verbose=verbose,
            )
        # return new object
        return fault

    def write_big_files(
        self,
        K_inner_asperities_file=None,
        K_inner_inner_file=None,
        cross_sections_file=None,
        verbose=False,
    ):
        """
        Write the arrays that are computationally expensive to files.

        Parameters
        ----------
        K_inner_asperities_file : str, optional
            File name for `K_inner_asperities`
        K_inner_inner_file : str, optional
            File name for `K_inner_inner`
        cross_sections_file : str, optional
            File name for `cross_sections`
        """
        if K_inner_asperities_file is not None:
            np.save(K_inner_asperities_file, self.K_inner_asperities)
            if verbose:
                print(f"Saved K_inner_asperities to '{K_inner_asperities_file}'")
        if K_inner_inner_file is not None:
            np.save(K_inner_inner_file, self.K_inner_inner)
            if verbose:
                print(f"Saved K_inner_inner to '{K_inner_inner_file}'")
        if cross_sections_file is not None:
            with open(cross_sections_file, "wb") as f:
                pickle.dump(self.cross_sections, f, pickle.HIGHEST_PROTOCOL)
                if verbose:
                    print(f"Saved cross_sections to '{cross_sections_file}'")

    @staticmethod
    def vectors_forward_rotation(vectors, R_from_to):
        """
        Rotate vectors from EFCS to another coordinate system.

        Parameters
        ----------
        vectors : numpy.ndarray
            Vectors arranged in the shape :math:`(n, 3)`
        R_from_to : numpy.ndarray
            Rotation matrices to use of shape :math:`(n, 3, 3)`

        Returns
        -------
        vectors_rot : numpy.ndarray
            Rotated vectors in the same shape.
        """
        return np.einsum("ijk,ik->ij", R_from_to, vectors)

    @staticmethod
    def vectors_backward_rotation(vectors, R_from_to):
        """
        Rotate vectors from another coordinate system to EFCS.

        Parameters
        ----------
        vectors : numpy.ndarray
            Vectors arranged in the shape :math:`(n, 3)`
        R_from_to : numpy.ndarray
            Rotation matrices to use of shape :math:`(n, 3, 3)`

        Returns
        -------
        vectors_rot : numpy.ndarray
            Rotated vectors in the same shape.
        """
        return np.einsum("ikj,ik->ij", R_from_to, vectors)

    @staticmethod
    def get_stress_kernel(
        patches, centroids, rot_mats_efcs, mu, nu, rot_mats_tdcs=None
    ):
        """
        Calculate the stress kernel between every source and target patch
        and project onto the target coordinate system.

        Parameters
        ----------
        patches : numpy.ndarray
            Source patches matrix of shape :math:`(n, 3, 3)`.
        centroids : numpy.ndarray
            Target centroids of shape :math:`(m, 3)`.
        rot_mats_efcs : numpy.ndarray
            Rotation matrices from EFCS to the target coordinate system
            of shape :math:`(m, 3, 3)`
        mu : float
            Shear modulus [Pa]
        nu : float
            Poisson's ratio [-]
        rot_mats_tdcs : numpy.ndarray, optional
            Rotation matrices from TDCS to the target coordinate system
            of shape :math:`(m, 3, 3)` (if the target is not TDCS)

        Returns
        -------
        K : numpy.ndarray
            Stress kernel (tensor) of shape :math:`(m, 3, n, 3)`.

        Notes
        -----

        To convert source slip in DDCS into target stress in DDCS all at once,
        use a tensor product: `numpy.tensordot(K, slip, axes=[(3, 2), (1, 0)])`.
        """

        # get dimensions and centroids
        num_source = patches.shape[0]
        num_target = centroids.shape[0]

        # get normal directions
        e1 = rot_mats_efcs[:, 0, :]
        e2 = rot_mats_efcs[:, 1, :]
        e3 = rot_mats_efcs[:, 2, :]

        # get stress tensor in EFCS for TDCS input
        strain_efcs = strain_matrix(obs_pts=centroids, tris=patches, nu=nu)
        stress_efcs = (
            strain_to_stress(
                strain=strain_efcs.transpose(0, 2, 3, 1).reshape(-1, 6), mu=mu, nu=nu
            )
            .reshape(num_target, num_source, 3, 6)
            .transpose(0, 3, 1, 2)
        )

        # change input coordinate system from TDCS to target one
        if rot_mats_tdcs is not None:
            stress_efcs = np.einsum(
                "ijkl,kml->ijkm", stress_efcs, rot_mats_tdcs, optimize="optimal"
            )

        # project to target coordinate system
        stress_efcs_3x3 = [
            np.stack(
                [
                    stress_efcs[:, 0, :, i].ravel(),
                    stress_efcs[:, 3, :, i].ravel(),
                    stress_efcs[:, 4, :, i].ravel(),
                    stress_efcs[:, 3, :, i].ravel(),
                    stress_efcs[:, 1, :, i].ravel(),
                    stress_efcs[:, 5, :, i].ravel(),
                    stress_efcs[:, 4, :, i].ravel(),
                    stress_efcs[:, 5, :, i].ravel(),
                    stress_efcs[:, 2, :, i].ravel(),
                ]
            )
            .reshape(3, 3, num_target, num_source)
            .transpose(2, 3, 0, 1)
            for i in range(3)
        ]
        path = np.einsum_path(
            "ik,ijkl,il->ij", e1, stress_efcs_3x3[0], e3, optimize="optimal"
        )[0]
        K = np.array(
            [
                [
                    np.einsum(
                        "ik,ijkl,il->ij", v, stress_efcs_3x3[i], e3, optimize=path
                    )
                    for i in range(3)
                ]
                for v in [e1, e2, e3]
            ]
        ).transpose(2, 0, 3, 1)

        # done
        return K

    @staticmethod
    def get_disp_kernel(obs_loc, patches, nu, rot_mats_tdcs=None):
        """
        Calculate the displacement kernel from a source mesh onto the observers.
        Defaults to TDCS input, can be changed by passing a rotation matrix.

        Parameters
        ----------
        obs_loc : numpy.ndarray
            Three-column EFCS locations of surface observers of shape :math:`(m, 3)`.
        patches : numpy.ndarray
            Source patches matrix of shape :math:`(n, 3, 3)`.
        nu : float
            Poisson's ratio [-]
        rot_mats_tdcs : numpy.ndarray, optional
            Rotation matrices from TDCS to the target coordinate system
            of shape :math:`(m, 3, 3)` (if the target is not TDCS)

        Returns
        -------
        G : numpy.ndarray
            Displacement kernel (tensor) of shape :math:`(m, 3, n, 3)`
            where :math:`m` is the number of observers and :math:`n` is the
            number of source patches.

        Notes
        -----

        To convert slip in TDCS into displacement in EFCS at all observers at once,
        use a tensor product: `numpy.tensordot(G, slip, axes=[(3, 2), (1, 0)])`.
        """

        # get displacement tensor from TDCS to EFCS
        G = disp_matrix(obs_pts=obs_loc, tris=patches, nu=nu)

        # change input coordinate system from TDCS to target one
        if rot_mats_tdcs is not None:
            G = np.einsum("ijkl,kml->ijkm", G, rot_mats_tdcs, optimize="optimal")

        # done
        return G

    @staticmethod
    def redistribute_normal(vectors):
        """
        Redistribute the normal (3rd) component onto the other directions, eliminating it
        but keeping the magnitude the same.

        Parameters
        ----------
        vectors : numpy.ndarray
            Vectors arranged in the shape :math:`(n, 3)`

        Returns
        -------
        vectors_rescaled : numpy.ndarray
            Rotated vectors in the shape :math:`(n, 2)`
        """
        ratio = np.sqrt(1 + (vectors[:, 2] / vectors[:, 1]) ** 2)
        vectors_rescaled = np.stack([vectors[:, 0], vectors[:, 1] * ratio], axis=1)
        assert np.allclose(
            np.linalg.norm(vectors, axis=1), np.linalg.norm(vectors_rescaled, axis=1)
        )
        return vectors_rescaled


class SubductionSimulation3D:
    """
    3D subduction simulation container class.
    """

    def __init__(
        self,
        n_cycles_max,
        n_samples_per_eq,
        slip_taper_distance,
        rheo,
        fault,
        D_0,
        D_0_logsigma,
        T_rec,
        T_rec_logsigma,
        T_anchor,
        T_last,
        enforce_v_plate,
        t_obs,
        pts_surf,
        atol=1e-8,
        rtol=1e-6,
        spinup_atol=1e-6,
        spinup_rtol=1e-3,
        v_max=None,
        calculate_tapered_slip=True,
        final_nonuniform_slip=None,
        G_surf=None,
        v_init=None,
        eq_df=None,
        alpha_h_vec=None,
        eq_slip=None,
        slip_taper_vec=None,
        slip_taper_vec_nonuni=None,
        delta_tau_unbounded=None,
        delta_tau_unbounded_nonuni=None,
        delta_tau_taper=None,
        delta_tau_taper_nonuni=None,
        locked_slip=None,
        delta_tau_bounded_compressed=None,
        delta_tau_bounded_indices=None,
    ):
        r"""
        Create a 3D subduction simulation.

        Parameters
        ----------
        n_cycles_max : int
            Maximum number of cycles to simulate [-]
        n_samples_per_eq : int
            Number of internal evaluation timesteps between earthquakes [-]
        slip_taper_distance : float
            Length scale over with to taper out the coseismic slip [m]
        rheo : Rheology
            Simulated upper plate interface's rheology.
        fault : Fault3D
            Fault object
        D_0 : numpy.ndarray
            Nominal coseismic displacement magnitude [m] of the locked asperities
            in the opposite direction of the plate velocity, dimensions (n_eq, n_asp)
        D_0_logsigma : numpy.ndarray
            Standard deviation of the displacement magnitude in logarithmic space,
            same shape as D_0
        T_rec : numpy.ndarray
            Nominal recurrence time [a] for each earthquake, length n_eq
        T_rec_logsigma : numpy.ndarray
            Standard deviation of the recurrence time in logarithmic space
        T_anchor : str
            Anchor date where observations end
        T_last : list
            Dates of the last occurence for each earthquake (list of strings)
        enforce_v_plate : bool
            Flag whether to allow the plate valocity in each patch to vary or not
            due to the random realizations of `D_0` and `T_rec`
        t_obs : numpy.ndarray, pandas.DatetimeIndex
            Observation timesteps, either as decimal years relative to the cycle start,
            or as Timestamps
        pts_surf : numpy.ndarray
            Observation coordinates [m] in EFCS, dimensions (n_stations, 2 or 3)
        atol : float
            Absolute tolerance of ODE integrator
        rtol : float
            Relative tolerance of ODE integrator
        spinup_atol : float
            Absolute tolerance of spinup controller
        spinup_rtol : float
            Relative tolerance of spinup controller
        v_max : float, optional
            Maximum slip velocity [m/s] on creeping patches during integration
        calculate_tapered_slip : bool, optional
            If ``True`` (default: ``False``), also calculate the assumed tapered
            slip implied by the coseismic stress changes
        final_nonuniform_slip : numpy.ndarray, optional
            A NumPy array that for all the observed earthquakes contains the amount of slip
            in all asperity patches [m] to be used after spinup.
        G_surf : numpy.ndarray, optional
            Skip the calculation of ``G_surf`` by passing it directly.
        v_init : numpy.ndarray, optional
            Skip the calculation of ``v_init`` by passing it directly.
        eq_df : pandas.DataFrame, optional
            Skip the calculation of ``eq_df`` by passing it directly.
        alpha_h_vec : numpy.ndarray, optional
            Skip the calculation of ``alpha_h_vec`` by passing it directly.
        eq_slip : numpy.ndarray, optional
            Skip the calculation of ``eq_slip`` by passing it directly.
        slip_taper_vec : numpy.ndarray, optional
            Skip the calculation of ``slip_taper_vec`` by passing it directly.
        slip_taper_vec_nonuni : numpy.ndarray, optional
            Skip the calculation of ``slip_taper_vec_nonuni`` by passing it directly.
        delta_tau_unbounded : numpy.ndarray, optional
            Skip the calculation of ``delta_tau_unbounded`` by passing it directly.
        delta_tau_unbounded_nonuni : numpy.ndarray, optional
            Skip the calculation of ``delta_tau_unbounded_nonuni`` by passing it directly.
        delta_tau_taper : numpy.ndarray, optional
            Skip the calculation of ``delta_tau_taper`` by passing it directly.
        delta_tau_taper_nonuni : numpy.ndarray, optional
            Skip the calculation of ``delta_tau_taper_nonuni`` by passing it directly.
        locked_slip : numpy.ndarray, optional
            Skip the calculation of ``locked_slip`` by passing it directly.
        delta_tau_bounded_compressed : numpy.ndarray, optional
            Skip the calculation of ``delta_tau_bounded_compressed`` by passing it directly.
        delta_tau_bounded_indices : numpy.ndarray, optional
            Skip the calculation of ``delta_tau_bounded_indices`` by passing it directly.
        """

        # save general sequence & fault parameters
        self.n_cycles_max = int(n_cycles_max)
        """ Maximum number of cycles to simulate [-] """
        self.n_samples_per_eq = int(n_samples_per_eq)
        """ Number of internal evaluation timesteps between earthquakes [-] """
        self.slip_taper_distance = float(slip_taper_distance)
        """ Length scale over with to taper out the coseismic slip [m] """
        self.atol = atol
        """ Absolute tolerance of ODE integrator """
        self.rtol = rtol
        """ Relative tolerance of ODE integrator """
        self.spinup_atol = spinup_atol
        """ Absolute tolerance of spinup controller """
        self.spinup_rtol = spinup_rtol
        """ Relative tolerance of spinup controller """
        self.v_max = None if v_max is None else float(v_max)
        """ Maximum slip velocity [m/s] on creeping patches during integration """

        # define rheology & fault
        assert isinstance(rheo, Rheology)
        assert isinstance(fault, Fault3D)
        self.rheo = rheo
        """ Simulated upper plate interface's Rheology object """
        self.fault = fault
        """ Fault object """

        # cast earthquake displacements as NumPy array
        self.D_0 = np.atleast_2d(D_0)
        """ Nominal coseismic displacement magnitudes [m] for each asperity (columns)
        and each earthquake (rows) """
        self.D_0_logsigma = np.atleast_2d(D_0_logsigma)
        """ Standard deviation of the displacement magnitudes in logarithmic space [-] """
        assert np.all(self.D_0 >= 0) and np.all(self.D_0_logsigma >= 0)
        assert self.D_0.shape == self.D_0_logsigma.shape
        assert self.D_0.shape[1] == self.fault.num_asperities
        # load recurrence times
        self.T_rec = np.atleast_1d(T_rec)
        """ Nominal recurrence time [a] for each earthquake """
        self.T_rec_logsigma = np.atleast_1d(T_rec_logsigma)
        """ Standard deviation of the recurrence time in logarithmic space """
        self.T_anchor = str(T_anchor)
        """ Anchor date where observations end """
        assert isinstance(T_last, list) and all([isinstance(tl, str) for tl in T_last])
        assert self.T_rec.size == self.T_rec_logsigma.size == self.D_0.shape[0]
        self.T_last = T_last
        """ Dates of the last occurence for each earthquake """
        # calculate how many earthquakes happen per cycle in each asperity
        self.T_fullcycle = np.lcm.reduce(self.T_rec)
        """ Nominal recurrence time [a] for an entire joint earthquake cycle """
        self.n_eq = self.D_0.shape[0]
        """ Number of distinct earthquakes in sequence """
        self.n_eq_per_cycle = (self.T_fullcycle / self.T_rec).astype(int)
        """ Number of earthquakes per full cycle """
        self.n_slips = self.n_eq_per_cycle.sum()
        """ Number of total earthquake events """
        total_cycle_clip_inferred_per_asperity = (
            self.D_0 * self.n_eq_per_cycle[:, None]
        ).sum(axis=0)
        v_plate_inferred = (
            total_cycle_clip_inferred_per_asperity / self.T_fullcycle / SEC_PER_YEAR
        )
        assert np.all(v_plate_inferred[0] == v_plate_inferred)
        self.v_plate_inferred = v_plate_inferred[0]
        """ Inferred plate velocity [m/s] based on the EQ setup """

        # check that in each asperity, the nominal plate rate is recovered
        # after a full earthquake cycle
        # this will fail once we have rotation-based plate velocity
        if self.fault.farfield_motion_type == Fault3D.FARFIELD_MOTION_UNIFORM:
            v_plate_vec_mag = np.linalg.norm(self.fault.v_plate_vec, axis=1)
            assert np.allclose(v_plate_vec_mag[0], v_plate_vec_mag)
            v_plate_mag = v_plate_vec_mag[0]
            total_cycle_slip_necessary = v_plate_mag * self.T_fullcycle * SEC_PER_YEAR
            assert np.allclose(
                total_cycle_clip_inferred_per_asperity, total_cycle_slip_necessary
            )

        # create realization of the slip amount and earthquake timings
        # first, create realizations of occurence times
        # note that this will result in a varying plate velocity rate
        rng = np.random.default_rng()
        self.T_rec_per_eq = [
            rng.lognormal(np.log(t), s, n)
            for t, s, n in zip(self.T_rec, self.T_rec_logsigma, self.n_eq_per_cycle)
        ]
        """ Recurrence time [a] realization """
        D_0_per_eq = [np.zeros((n, self.D_0.shape[1])) for n in self.n_eq_per_cycle]
        for i, (d, s, n) in enumerate(
            zip(self.D_0, self.D_0_logsigma, self.n_eq_per_cycle)
        ):
            d_nonzero = d > 0
            D_0_per_eq[i][:, d_nonzero] = rng.lognormal(
                np.log(d[d_nonzero]), s[d_nonzero], (n, (d_nonzero).sum())
            )
        self.D_0_per_eq = D_0_per_eq
        """ Coseismic slip magnitude [m] realization """

        # second, we need to shift the random realization for each earthquake
        # individually such that they all yield the same v_plate_mag (enforced or not)
        # adjust that all asperities slip the same amount over an entire cycle
        D_0_per_eq_concat = np.concatenate(self.D_0_per_eq, axis=0)
        D_0_per_asp = D_0_per_eq_concat.sum(axis=0, keepdims=True)
        D_0_per_asp_mean = D_0_per_asp.mean()
        D_0_per_eq_concat_adj = D_0_per_eq_concat * D_0_per_asp_mean / D_0_per_asp
        D_0_per_eq_adj = np.split(
            D_0_per_eq_concat_adj, np.cumsum(self.n_eq_per_cycle)[:-1], axis=0
        )
        # adjust all individual recurrence times such that they have the same overall
        T_fullcycle_concat = np.array([t.sum() for t in self.T_rec_per_eq])
        T_fullcycle_mean = T_fullcycle_concat.mean()
        T_rec_per_eq_adj = [t * T_fullcycle_mean / t.sum() for t in self.T_rec_per_eq]
        # now each asperity has the same effective plate velocity, which can be different
        # from the nominal one - if we want to enforce the nominal plate velocity,
        # we can rescale the recurrence times again
        self.enforce_v_plate = bool(enforce_v_plate)
        """ Flag whether to allow v_plate_mag to vary or not """
        # get implied, effective plate velocity
        v_plate_eff_mag = (
            np.concatenate(D_0_per_eq_adj, axis=0).sum(axis=0)
            / np.array([t.sum() for t in T_rec_per_eq_adj])[:, None]
            / SEC_PER_YEAR
        ).ravel()
        assert np.allclose(v_plate_eff_mag[0], v_plate_eff_mag)
        # only need to keep single value
        v_plate_eff_mag = v_plate_eff_mag[0]
        if self.enforce_v_plate:
            v_plate_factor = self.v_plate_inferred / v_plate_eff_mag
            for i in range(self.n_eq):
                T_rec_per_eq_adj[i] /= v_plate_factor
            v_plate_eff_mag = self.v_plate_inferred
        self.v_plate_eff_mag = v_plate_eff_mag
        """ Effective far-field plate velocity [m/s] """
        self.T_eff = T_rec_per_eq_adj[0].sum()
        """ Effective length [a] of entire earthquake sequence """
        # calculate plate velocity vector on all patches,
        # with the lower interface going the opposite direction
        v_scale_ratio = self.v_plate_eff_mag / self.v_plate_inferred
        self.v_plate_vec_eff = self.fault.v_plate_vec * v_scale_ratio
        """ Effective plate velocity in 3D in all patches """
        self.v_plate_vec_eff_mag = np.linalg.norm(self.v_plate_vec_eff, axis=1)
        """ Effective plate velocity magnitude in all patches """
        # project the plate velocity onto the DDCS
        self.v_plate_ddcs_proj_eff = (
            self.fault.e_plate_ddcs_proj * self.v_plate_vec_eff_mag[:, None]
        )
        """ Projected, effective plate velocity in DDCS """

        # third, we need to create a list of earthquake dates and associated slips
        if eq_df is None:
            year_offsets = [
                (pd.Period(self.T_anchor, "D") - pd.Period(self.T_last[i], "D")).n
                / 365.25
                for i in range(self.n_eq)
            ]
            eq_df_index = np.concatenate(
                [
                    self.T_eff
                    - (
                        np.cumsum(T_rec_per_eq_adj[i])
                        - T_rec_per_eq_adj[i]
                        + year_offsets[i]
                    )
                    for i in range(self.n_eq)
                ]
            )
            # round the dates to the closest day and combine earthquakes
            eq_df_index_rounded = np.around(eq_df_index * 365.25) / 365.25
            # build a DataFrame with exact and rounded times
            eq_df = pd.DataFrame(data=np.concatenate(D_0_per_eq_adj, axis=0))
            eq_df["time"] = eq_df_index
            eq_df["rounded"] = eq_df_index_rounded
            # now aggregate by rounded time, keeping the minimum exact time, and summing slip
            agg_dict = {"time": "min"}
            agg_dict.update({c: "sum" for c in range(self.D_0.shape[1])})
            eq_df = eq_df.groupby("rounded").agg(agg_dict)
            # convert time column to index and sort
            eq_df.set_index("time", inplace=True)
            eq_df.sort_index(inplace=True)
            assert np.allclose(eq_df.sum(axis=0), eq_df.sum(axis=0)[0])
        self.eq_df = eq_df
        """
        DataFrame with the dates [decimal year from cycle start] and slips [m]
        for each asperity based on inferred plate velocity
        """

        # fourth, we need to create a list of dates to use internally when evaluating
        # the earthquake cycle - this is independent of the observation dates
        i_frac_cumsum = np.concatenate(
            [[self.eq_df.index[-1] - self.T_eff], self.eq_df.index.values]
        )
        T_frac = np.diff(i_frac_cumsum)
        logspace = np.logspace(
            0, np.log10(1 + T_frac), self.n_samples_per_eq, endpoint=False
        )
        p1 = np.arange(self.n_cycles_max).reshape(-1, 1, 1) * self.T_eff
        p2 = i_frac_cumsum[:-1].reshape(1, -1, 1)
        p3 = logspace.T[None, :, :]
        t_eval = (p1 - 1 + p2 + p3).ravel()
        num_neg = (t_eval < 0).sum()
        t_eval = np.roll(t_eval, -num_neg)
        t_eval[-num_neg:] += self.n_cycles_max * self.T_eff
        self.t_eval = np.sort(
            np.concatenate([t_eval, np.arange(self.n_cycles_max + 1) * self.T_eff])
        )
        """ Internal evaluation timesteps [decimal years since cycle start] """
        self.n_eval = self.t_eval.size
        """ Number of internal evaluation timesteps [-] """

        # fifth, for the integration, we need the indices of the timesteps that mark either
        # an earthquake or the start of a new cycle
        self.n_slips = self.eq_df.shape[0]
        """ Number of slips in a sequence [-] """
        self.ix_break = [
            i * (self.n_slips * self.n_samples_per_eq + 1)
            for i in range(self.n_cycles_max + 1)
        ]
        """ Indices of breaks between cycles """
        p1 = np.asarray(self.ix_break)[: self.n_cycles_max, None]
        p2 = np.arange(1, 1 + self.n_slips)[None, :]
        self.ix_eq = (p1 + p2 * self.n_samples_per_eq - num_neg + 1).ravel().tolist()
        """ Indices of earthquakes """

        # sixth and last, for the final loop, we need a joint timesteps array between internal
        # and external (observation) timestamps, such that we can debug, check early stopping,
        # and restrict the output to the requested timeseries
        if isinstance(t_obs, pd.DatetimeIndex):
            t_obs_dates = t_obs.values
            t_obs = (
                self.T_eff
                + (t_obs - pd.Timestamp(self.T_anchor)).total_seconds().values
                / SEC_PER_YEAR
            )
        elif isinstance(t_obs, np.ndarray):
            if np.issubdtype(t_obs.dtype, np.datetime64):
                t_obs_dates = t_obs
                t_obs = (
                    self.T_eff
                    + (pd.DatetimeIndex(t_obs) - pd.Timestamp(self.T_anchor))
                    .total_seconds()
                    .values
                    / SEC_PER_YEAR
                )
            elif np.all(t_obs <= 0):
                # this format is relative to T_anchor and more stable when T_eff varies
                t_obs_dates = pd.Timestamp(self.T_anchor) + t_obs * pd.Timedelta(
                    365.25, "D"
                )
                t_obs = self.T_eff + t_obs
            assert np.all(t_obs >= 0) and np.all(t_obs < self.T_eff), (
                f"Range of 't_obs' ({t_obs.min()}-{t_obs.max():} years) outside of "
                f"the earthquake cycle period ({self.T_eff:} years)."
            )
        else:
            raise ValueError("Unknown 't_obs' data type.")
        self.t_obs = t_obs
        """ Observation timesteps [decimal years since cycle start] """
        self.t_obs_dates = t_obs_dates
        """ Observation timesteps [datetime format] """
        # combine all possible timesteps
        t_obs_shifted = self.t_obs + (self.n_cycles_max - 1) * self.T_eff
        t_eval_joint = np.concatenate((self.t_eval, t_obs_shifted))
        # round to nearest second for numerical stability later
        t_eval_joint = np.round(t_eval_joint * SEC_PER_YEAR) / SEC_PER_YEAR
        self.t_eval_joint = np.unique(t_eval_joint)
        """
        Joint internal evaluation and external observation timesteps
        [decimal years since cycle start]
        """
        self.t_eval_obs = pd.Timestamp(self.T_anchor) + (
            self.t_eval[
                np.logical_and(
                    self.t_eval - self.T_eff * (self.n_cycles_max - 1)
                    >= self.t_obs.min(),
                    self.t_eval - self.T_eff * (self.n_cycles_max - 1)
                    <= self.t_obs.max(),
                )
            ]
            - self.T_eff * self.n_cycles_max
        ) * pd.Timedelta(365.25, "D")
        """
        Timestamps of the internal evaluation timesteps that fall within
        the external observation time window
        """
        # get indices of each individual subset in the new timesteps array
        ix_break_joint = np.isin(
            self.t_eval_joint,
            np.round(self.t_eval[self.ix_break] * SEC_PER_YEAR) / SEC_PER_YEAR,
        )
        self.ix_break_joint = np.flatnonzero(ix_break_joint)
        """ Indices of breaks between cycles in joint timesteps """
        ix_eq_joint = np.isin(
            self.t_eval_joint,
            np.round(self.t_eval[self.ix_eq] * SEC_PER_YEAR) / SEC_PER_YEAR,
        )
        self.ix_eq_joint = np.flatnonzero(ix_eq_joint)
        """ Indices of earthquakes in joint timesteps """
        ix_obs_joint = np.isin(
            self.t_eval_joint, np.round(t_obs_shifted * SEC_PER_YEAR) / SEC_PER_YEAR
        )
        self.ix_obs_joint = np.flatnonzero(ix_obs_joint)
        """ Indices of observation timestamps in joint timesteps """
        assert not np.any(np.logical_and(ix_break_joint, ix_eq_joint))
        assert ix_break_joint.sum() == len(self.ix_break)
        assert ix_eq_joint.sum() == len(self.ix_eq)
        assert ix_obs_joint.sum() == self.t_obs.size

        # get vectors of upper plate alpha_h
        if alpha_h_vec is None:
            alpha_h_vec, dominant_bases = self.rheo.get_param_vectors(
                self.fault.inner_patch_2d_coords[:, 0],
                self.fault.inner_patch_2d_coords[:, 1],
                self.fault.inner_centroids,
                self.fault.outer_vertices,
            )
            alpha_h_vec = alpha_h_vec[:, None]
        else:
            dominant_bases = None
        self.alpha_h_vec = alpha_h_vec
        r""" Depth-variable :math:`\alpha_h` [Pa] of upper plate interface """
        self.dominant_bases = dominant_bases
        """ Index of which basis function is strongest at each patch [-] """

        # get indices of first timestep after earthquake (for resets to zero)
        self.slips_obs = np.logical_and(
            self.t_obs.min() <= self.eq_df.index, self.t_obs.max() > self.eq_df.index
        )
        """ Mask of which earthquakes in a cycle are observed """
        self.n_slips_obs = self.slips_obs.sum()
        """ Number of observed earthquakes [-] """
        if self.n_slips_obs > 0:
            i_slips_obs = [
                np.argmax(self.t_obs >= t_eq)
                for t_eq in self.eq_df.index.values[self.slips_obs]
            ]
        else:
            i_slips_obs = None
        self.i_slips_obs = i_slips_obs
        """ Indices in t_obs for first observation after earthquakes [-] """

        # project eq_df onto each patch using the directions of v_plate to get slip
        if eq_slip is None:
            total_slip_eff_asp = self.eq_df.sum(axis=0).values
            assert np.allclose(total_slip_eff_asp[0] - total_slip_eff_asp, 0)
            total_slip_from_v_plate = (
                self.v_plate_vec_eff_mag * SEC_PER_YEAR * self.T_eff
            )
            total_slip_from_v_plate_asp = total_slip_from_v_plate[
                self.fault.s_asperities
            ]
            if self.fault.farfield_motion_type == Fault3D.FARFIELD_MOTION_UNIFORM:
                assert np.allclose(
                    total_slip_from_v_plate[0] - total_slip_from_v_plate, 0
                )
            eq_df_frac = self.eq_df / self.T_eff / self.v_plate_inferred / SEC_PER_YEAR
            asp_eq_slip = [
                eq_df_frac.values[:, jasp][:, None]
                * total_slip_from_v_plate_asp[
                    self.fault.asperity_indices[jasp][None, :]
                ]
                for jasp in range(self.fault.num_asperities)
            ]
            eq_slip = np.concatenate(asp_eq_slip, axis=1)
        self.eq_slip = eq_slip
        """ Coseismic slip magnitude for each earthquake in each asperity [m] """
        self.eq_ddcs_proj = (
            self.eq_slip[:, :, None]
            * self.fault.e_plate_ddcs_proj[None, self.fault.s_asperities, :]
        )
        """ Coseismic slip for each earthquake in each asperity in DDCS [m] """

        # calculate slip tapering
        if (slip_taper_vec is None) or (
            (final_nonuniform_slip is not None) and (slip_taper_vec_nonuni is None)
        ):
            # evaluate tapering function
            dist_vals_scaled = 1 - erf(
                self.fault.dists_inner_asperities / self.slip_taper_distance
            )
            # subset for the closest patches
            dist_vals_scaled_closest = np.take_along_axis(
                dist_vals_scaled, self.fault.i_closest_asp_tri, axis=0
            )

        # apply to uniform events
        if slip_taper_vec is None:
            # get the slip magnitude in the relevant asperity triangles
            rel_asp_slip = self.eq_slip[
                :, self.fault.i_closest_asp_tri.ravel()
            ].reshape(
                self.n_slips, self.fault.num_asperities, self.fault.inner_num_patches
            )
            # get the slip taper for each EQ, asperity, and patch
            slip_taper_asp = rel_asp_slip * dist_vals_scaled_closest[None, :, :]
            # get the maximum
            slip_taper_mag = np.max(slip_taper_asp, axis=1)
            slip_taper_vec = (
                slip_taper_mag[:, :, None]
                * self.fault.e_plate_ddcs_proj[None, fault.s_inner, :]
            )
        self.slip_taper_vec = slip_taper_vec
        """ Tapered slip vector in the inner mesh for each earthquake in DDCS [m] """

        # get unbounded delta_tau
        if delta_tau_unbounded is None:
            delta_tau_unbounded = np.tensordot(
                self.fault.K_inner_asperities[:, :2, :, :2],
                self.eq_ddcs_proj,
                axes=[(3, 2), (2, 1)],
            ).transpose(2, 0, 1)
        self.delta_tau_unbounded = delta_tau_unbounded
        """ Array of unbounded coseismic stress changes [Pa] """
        if delta_tau_taper is None:
            delta_tau_taper = np.tensordot(
                self.fault.K_inner_inner[:, :2, :, :2],
                self.slip_taper_vec,
                axes=[(3, 2), (2, 1)],
            ).transpose(2, 0, 1)
        self.delta_tau_taper = delta_tau_taper
        """ Array of taper-induced coseismic stress changes [Pa] """
        self.delta_tau_bounded = self.delta_tau_unbounded + self.delta_tau_taper
        """ Bounded (tapered) coseismic stress change [Pa] """

        # generate a copy of delta_tau_bounded with non-uniform slip, if provided
        if final_nonuniform_slip is not None:
            # check shape
            assert isinstance(
                final_nonuniform_slip, np.ndarray
            ), "'final_nonuniform_slip' needs to be a NumPy array"
            assert (
                final_nonuniform_slip.shape[1] == self.fault.asperities_num_patches
            ), (
                "'final_nonuniform_slip' needs to be have a exactly "
                f"{self.fault.asperities_num_patches} columns, got "
                f"{final_nonuniform_slip.shape[1]}."
            )
            n_nonuni_slips = final_nonuniform_slip.shape[0]
            self.eq_ddcs_proj_nonuni = (
                final_nonuniform_slip[:, :, None]
                * self.fault.e_plate_ddcs_proj[None, self.fault.s_asperities, :]
            )
            """ Non-uniform slip for each observed earthquake and asperity in DDCS [m] """
            # calculate taper
            if slip_taper_vec_nonuni is None:
                rel_asp_slip_nonuni = final_nonuniform_slip[
                    :, self.fault.i_closest_asp_tri.ravel()
                ].reshape(
                    n_nonuni_slips,
                    self.fault.num_asperities,
                    self.fault.inner_num_patches,
                )
                slip_taper_asp_nonuni = (
                    rel_asp_slip_nonuni * dist_vals_scaled_closest[None, :, :]
                )
                slip_taper_mag_nonuni = np.max(slip_taper_asp_nonuni, axis=1)
                slip_taper_vec_nonuni = (
                    slip_taper_mag_nonuni[..., None]
                    * self.fault.e_plate_ddcs_proj[None, self.fault.s_inner, :]
                )
            self.slip_taper_vec_nonuni = slip_taper_vec_nonuni
            """ Non-uniform tapered slip for each observed earthquake and asperity in DDCS [m] """
            # calculate stress changes
            if delta_tau_unbounded_nonuni is None:
                delta_tau_unbounded_nonuni = np.tensordot(
                    self.fault.K_inner_asperities[:, :2, :, :2],
                    self.eq_ddcs_proj_nonuni,
                    axes=[(3, 2), (2, 1)],
                ).transpose(2, 0, 1)
            self.delta_tau_unbounded_nonuni = delta_tau_unbounded_nonuni
            """ Array of unbounded non-uniform stress changes [Pa] """
            if delta_tau_taper_nonuni is None:
                delta_tau_taper_nonuni = np.tensordot(
                    self.fault.K_inner_inner[:, :2, :, :2],
                    self.slip_taper_vec_nonuni,
                    axes=[(3, 2), (2, 1)],
                ).transpose(2, 0, 1)
            self.delta_tau_taper_nonuni = delta_tau_taper_nonuni
            """ Array of taper-induced non-uniform coseismic stress changes [Pa] """
            delta_tau_bounded_nonuni = (
                self.delta_tau_unbounded_nonuni + self.delta_tau_taper_nonuni
            )
        else:
            delta_tau_bounded_nonuni = None
        self.delta_tau_bounded_nonuni = delta_tau_bounded_nonuni
        """ Bounded non-uniform stress change [Pa] """

        # get the additional slip during the observed period
        if calculate_tapered_slip:
            slip_taper = self.slip_taper_vec.copy()
            if final_nonuniform_slip is not None:
                slip_taper[-n_nonuni_slips:, :, :] = self.slip_taper_vec_nonuni
            slip_taper_strike = (
                pd.DataFrame(index=self.eq_df.index, data=slip_taper[:, :, 0])
                .cumsum(axis=0)
                .reindex(index=self.t_obs, method="ffill", fill_value=0)
                .values.T
            )
            slip_taper_dip = (
                pd.DataFrame(index=self.eq_df.index, data=slip_taper[:, :, 1])
                .cumsum(axis=0)
                .reindex(index=self.t_obs, method="ffill", fill_value=0)
                .values.T
            )
            self.slip_taper = np.stack([slip_taper_strike, slip_taper_dip], axis=1)
            """ Timeseries of tapered slip [m] on the upper creeping fault patches """

        # create the Green's matrices
        pts_surf = np.atleast_2d(pts_surf)
        assert (pts_surf.ndim == 2) and (pts_surf.shape[1] in [2, 3])
        if pts_surf.shape[1] == 2:
            pts_surf = np.concatenate(
                [pts_surf, np.zeros((pts_surf.shape[0], 1))], axis=1
            )
        self.pts_surf = np.ascontiguousarray(pts_surf)
        """ Coordinates of surface observation points [m] """
        self.n_observers = self.pts_surf.shape[0]
        """ Number of observers """
        if G_surf is None:
            print("Calculating displacement kernel... ", flush=True, end="")
            tic = perf_counter()
            G_surf = Fault3D.get_disp_kernel(
                self.pts_surf,
                self.fault.joint_patches,
                self.fault.nu,
                rot_mats_tdcs=self.fault.R_joint_tdcs_to_ddcs,
            )[:, :, :, :2]
            toc = perf_counter()
            print(f"done ({toc - tic:.2f} s)", flush=True)
        else:
            assert isinstance(
                G_surf, np.ndarray
            ), f"Expected 'G_surf' NumPy array, got {type(G_surf)}."
            expected_shape = (self.n_observers, 3, self.fault.joint_num_patches, 2)
            assert (
                G_surf.shape == expected_shape
            ), f"Expected 'G_surf' shape of {expected_shape}, got {G_surf.shape}."
            print(f"Accepted input 'G_surf' of shape {expected_shape}.")
        self.G_surf = G_surf
        """ Green's matrix [-] relating slip on all patches to surface motion """

        # get slip timeseries for the non-simulated patches
        if final_nonuniform_slip is None:
            eq_ddcs_proj_for_obs = self.eq_ddcs_proj
        else:
            eq_ddcs_proj_for_obs = self.eq_ddcs_proj.copy()
            eq_ddcs_proj_for_obs[-n_nonuni_slips:, :, :] = self.eq_ddcs_proj_nonuni
        if locked_slip is None:
            locked_slip_strike = (
                pd.DataFrame(index=self.eq_df.index, data=eq_ddcs_proj_for_obs[:, :, 0])
                .cumsum(axis=0)
                .reindex(index=self.t_obs, method="ffill", fill_value=0)
                .values.T
            )
            locked_slip_dip = (
                pd.DataFrame(index=self.eq_df.index, data=eq_ddcs_proj_for_obs[:, :, 1])
                .cumsum(axis=0)
                .reindex(index=self.t_obs, method="ffill", fill_value=0)
                .values.T
            )
            locked_slip = np.stack([locked_slip_strike, locked_slip_dip], axis=1)
        self.locked_slip = locked_slip
        """ Timeseries of slip [m] on the locked patches for observation timespan """
        self.outer_creep_slip = (
            self.t_obs[None, None, :]
            * SEC_PER_YEAR
            * self.v_plate_ddcs_proj_eff[self.fault.s_outer, :, None]
        )
        """ Timeseries of slip [m] on the outer creep patches for observation timestamps """
        self.lower_creep_slip = (
            self.t_obs[None, None, :]
            * SEC_PER_YEAR
            * self.v_plate_ddcs_proj_eff[self.fault.s_lower, :, None]
        )
        """ Timeseries of slip [m] on the lower creep patches for observation timestamps """

        # pre-calculate the constant stressing rate from the asperities
        self.K_inner_asperities_v_plate = np.tensordot(
            self.fault.K_inner_asperities[:, :2, :, :2],
            self.v_plate_ddcs_proj_eff[self.fault.s_asperities, :],
            axes=[(3, 2), (1, 0)],
        )
        """ Stressing rate from the locked asperities onto the simulated patches [Pa/s] """

        # calculate initial velocity from steady-state stress balance
        if v_init is None:
            v_plate = self.v_plate_ddcs_proj_eff[self.fault.s_inner, :].ravel()
            K_ext_v_plate = self.K_inner_asperities_v_plate.ravel()
            K_int = self.fault.K_inner_inner[:, :2, :, :2].reshape(
                v_plate.size, v_plate.size
            )
            v_init = np.linalg.lstsq(
                K_int, K_int @ v_plate + K_ext_v_plate, rcond=None
            )[0].reshape(self.fault.inner_num_patches, 2)
        else:
            expected_shape = (self.fault.inner_num_patches, 2)
            assert v_init.shape == expected_shape, (
                "Couldn't load initial velocities due to shape mismatch:\n"
                f"Loaded = {v_init.shape}, expected = {expected_shape}."
            )
        self.v_init = v_init
        """ Initial velocity in all creeping patches [m/s] """

        # get compressed version of delta_tau_bounded that only has unique events
        if (delta_tau_bounded_compressed is None) or (
            delta_tau_bounded_indices is None
        ):
            delta_tau_bounded_compressed, delta_tau_bounded_indices = np.unique(
                self.delta_tau_bounded, axis=0, return_inverse=True
            )
        self.delta_tau_bounded_compressed = delta_tau_bounded_compressed
        """ Bounded coseismic stress change [Pa] for unique events """
        self.delta_tau_bounded_indices = delta_tau_bounded_indices
        """ Indices to convert the compressed coseismic stress changes into the full ones [-] """

    def update_rheo(self, rheo):
        """
        Update the current simulation object to match a new rheology, keeping everything else
        the same.

        Parameters
        ----------
        rheo : Rheology
            New rheology object.
        """
        # save rheology object
        assert isinstance(rheo, Rheology)
        self.rheo = rheo
        # update alpha_h_vec
        alpha_h_vec, dominant_bases = self.rheo.get_param_vectors(
            self.fault.inner_patch_2d_coords[:, 0],
            self.fault.inner_patch_2d_coords[:, 1],
            self.fault.inner_centroids,
            self.fault.outer_vertices,
        )
        alpha_h_vec = alpha_h_vec[:, None]
        self.alpha_h_vec = alpha_h_vec
        self.dominant_bases = dominant_bases
        # done

    def get_vels(self, state):
        """
        Calculate the observed surface velocities from different contributions.

        Parameters
        ----------
        state : numpy.ndarray
            State variable of fault interface.

        Returns
        -------
        surf_vels_sim : numpy.ndarray
            Surface velocities [m/s] due to the simulated part of the fault interface.
        surf_vels_outer : numpy.ndarray
            Surface velocities [m/s] due to the outer part of the fault interface.
        surf_vels_lower : numpy.ndarray
            Surface velocities [m/s] due to the lower side of the fault interface.
        """
        surf_vels_sim = get_surface_displacements(
            state[self.fault.inner_num_patches :, :, :],
            self.G_surf[:, :, self.fault.s_inner, :],
        )
        outer_creep_vel = self.v_plate_ddcs_proj_eff[self.fault.s_outer, :, None]
        surf_vels_outer = get_surface_displacements(
            outer_creep_vel, self.G_surf[:, :, self.fault.s_outer, :]
        )
        lower_creep_vel = self.v_plate_ddcs_proj_eff[self.fault.s_lower, :, None]
        surf_vels_lower = get_surface_displacements(
            lower_creep_vel, self.G_surf[:, :, self.fault.s_lower, :]
        )
        return surf_vels_sim, surf_vels_outer, surf_vels_lower

    def get_euler_pole_kernel(self):
        """
        Calculate the displacement kernel that applies an Euler pole rotation
        to the observers.

        Returns
        -------
        G_ep : numpy.ndarray
            Euler pole displacement kernel of shape :math:`(m, 2, 3)`,
            where :math:`m` is the number of observers.
        """
        assert self.fault.utmzone is not None
        # calculate observer locations in ECEF and LLA coordinates
        crs_lla = ccrs.Geodetic()
        crs_utm = ccrs.UTM(zone=self.fault.utmzone)
        crs_xyz = ccrs.Geocentric()
        loc_obs_lla = crs_lla.transform_points(
            crs_utm, self.pts_surf[:, 0], self.pts_surf[:, 1]
        )
        loc_obs_xyz = crs_xyz.transform_points(
            crs_utm, self.pts_surf[:, 0], self.pts_surf[:, 1]
        )
        # create kernel
        R_cross_obs = [crossmat(x1, x2, x3) for x1, x2, x3 in loc_obs_xyz]
        R_ecef2enu_obs = [R_ecef2enu(lo, la) for lo, la in loc_obs_lla[:, :2]]
        G_ep = np.stack(
            [
                (R_ecef2enu_obs[i] @ R_cross_obs[i])[:2, :]
                for i in range(self.pts_surf.shape[0])
            ]
        )
        # done
        return G_ep

    @staticmethod
    def read_config_file(config_file):
        """
        Read a configuration file and return it as parsed dictionaries.

        Parameters
        ----------
        config_file : str
            Path to INI configuration file.

        Returns
        -------
        rheo_dict : dict
            Parsed configuration file for the rheology object.
        fault_dict : dict
            Parsed configuration file for the fault object.
        sim_dict : dict
            Parsed configuration file for the simulation object.
        """

        # load configuration file
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        with open(config_file, mode="rt") as f:
            cfg.read_file(f)
        cfg_sim, cfg_fault, cfg_rheo = cfg["simulation"], cfg["fault"], cfg["rheology"]

        # create rheology dictionary
        rheo_dict = {
            "v_0": cfg_rheo.getfloat("v_0"),
            "alpha_h_mat": np.atleast_2d(json.loads(cfg_rheo["alpha_h_mat"])),
            "degree": cfg_rheo.getint("degree"),
            "boundary_width": cfg_rheo.getfloat("boundary_width"),
            "alpha_h_boundary": cfg_rheo.getfloat("alpha_h_boundary"),
        }

        # prepare fault dictionary
        farfield_motion_type = cfg_fault.getint("farfield_motion_type")
        if farfield_motion_type == Fault3D.FARFIELD_MOTION_PROVIDED:
            farfield_motion = np.load(cfg_fault["farfield_motion"])
        else:
            farfield_motion = np.atleast_1d(json.loads(cfg_fault["farfield_motion"]))
        # load custom stress kernels
        K_inner_asperities_file = cfg_fault.get("K_inner_asperities_file")
        K_inner_asperities = (
            None
            if K_inner_asperities_file is None
            else np.load(K_inner_asperities_file)
        )
        K_inner_inner_file = cfg_fault.get("K_inner_inner_file")
        K_inner_inner = (
            None if K_inner_inner_file is None else np.load(K_inner_inner_file)
        )

        # create fault dictionary
        fault_dict = {
            "inner_mesh_files": json.loads(cfg_fault["inner_mesh_files"]),
            "asperity_mesh_files": json.loads(cfg_fault["asperity_mesh_files"]),
            "lower_mesh_files": json.loads(cfg_fault["lower_mesh_files"]),
            "downdip_direction": cfg_fault.get("downdip_direction"),
            "mu": cfg_fault.getfloat("mu"),
            "nu": cfg_fault.getfloat("nu"),
            "v_s": cfg_fault.getfloat("v_s"),
            "farfield_motion": farfield_motion,
            "farfield_motion_type": farfield_motion_type,
            "max_simulated_distance": cfg_fault.getfloat("max_simulated_distance"),
            "max_simulated_depth": cfg_fault.getfloat("max_simulated_depth"),
            "orient_along": cfg_fault.getfloat("orient_along"),
            "utmzone": cfg_fault.getint("utmzone"),
            "lon0": cfg_fault.getfloat("lon0"),
            "lat0": cfg_fault.getfloat("lat0"),
            "offset_lower": cfg_fault.getfloat("offset_lower", 0),
            "K_inner_asperities": K_inner_asperities,
            "K_inner_inner": K_inner_inner,
        }

        # prepare simulation dictionary
        # load earthquake setup file
        eq_setup = pd.read_csv(cfg_sim.get("eq_file"), index_col=0)
        assert (eq_setup.shape[1] - 3) % 2 == 0
        num_asperities = (eq_setup.shape[1] - 3) // 2
        T_last = eq_setup["T_last"].tolist()
        T_rec = eq_setup["T_rec"].values
        T_rec_logsigma = eq_setup["T_rec_logsigma"].values
        D_0 = eq_setup[[f"D_0_{i}" for i in range(num_asperities)]].values
        D_0_logsigma = eq_setup[
            [f"D_0_logsigma_{i}" for i in range(num_asperities)]
        ].values
        # load observation timestamps
        t_obs = np.load(cfg_sim.get("t_obs_file"))
        # load observer locations
        obs_loc = pd.read_csv(cfg_sim.get("obs_loc_file"), index_col=0)
        pts_surf = obs_loc.values
        # load final nonuniform slip file
        final_nonuniform_slip_file = cfg_sim.get("final_nonuniform_slip_file")
        final_nonuniform_slip = (
            None
            if final_nonuniform_slip_file is None
            else np.load(final_nonuniform_slip_file)
        )
        # load initial velocity file
        v_init_file = cfg_sim.get("v_init_file")
        v_init = None if v_init_file is None else np.load(v_init_file)
        # load custom displacement kernel
        G_surf_file = cfg_sim.get("G_surf_file")
        G_surf = None if G_surf_file is None else np.load(G_surf_file)

        # create simulation dictionary
        sim_dict = {
            "n_cycles_max": cfg_sim.getint("n_cycles_max"),
            "n_samples_per_eq": cfg_sim.getint("n_samples_per_eq"),
            "slip_taper_distance": cfg_sim.getfloat("slip_taper_distance"),
            "D_0": D_0,
            "D_0_logsigma": D_0_logsigma,
            "T_rec": T_rec,
            "T_rec_logsigma": T_rec_logsigma,
            "T_anchor": cfg_sim.get("T_anchor"),
            "T_last": T_last,
            "enforce_v_plate": cfg_sim.getboolean("enforce_v_plate"),
            "t_obs": t_obs,
            "pts_surf": pts_surf,
            "atol": cfg_sim.getfloat("atol"),
            "rtol": cfg_sim.getfloat("rtol"),
            "spinup_atol": cfg_sim.getfloat("spinup_atol"),
            "spinup_rtol": cfg_sim.getfloat("spinup_rtol"),
            "v_max": cfg_sim.getfloat("v_max"),
            "final_nonuniform_slip": final_nonuniform_slip,
            "G_surf": G_surf,
            "v_init": v_init,
        }

        # done
        return rheo_dict, fault_dict, sim_dict

    @classmethod
    def from_cfg_objs_and_files(
        cls,
        sim_dict,
        rheo,
        fault,
        G_surf_file=None,
        write_new_files=True,
        verbose=False,
    ):
        """
        Create a `SubductionSimulation3D` object from a configuration dictionary,
        precomputed objects and files.

        Parameters
        ----------
        sim_dict : dict
            Dictionary of keyword arguments passed on to the initialization; from
            `SubductionSimulation3D.read_config_file`
        rheo : Rheology
            Simulated upper plate interface's rheology.
        fault : Fault3D
            Fault object
        G_surf_file : str, optional
            File name for `G_surf`
        write_new_files : bool, optional
            Whether to save to files the kernels and cross sections if they were missing
        verbose : bool, optional
            Print what's happening

        Returns
        -------
        sim : Fault3D
            The new `Fault3D` object
        """
        # check for G_surf
        if G_surf_file is not None:
            try:
                G_surf = np.load(G_surf_file)
            except FileNotFoundError:
                G_surf = None
                if verbose:
                    print(f"Couldn't find '{G_surf_file}', " "need to recompute G_surf")
            else:
                if verbose:
                    print(f"Loaded G_surf from '{G_surf_file}'")
            if "G_surf" in sim_dict:
                del sim_dict["G_surf"]
        else:
            G_surf = sim_dict.pop("G_surf", None)
        # create new object
        sim = cls(**sim_dict, rheo=rheo, fault=fault, G_surf=G_surf)
        # save new files if desired
        if write_new_files:
            if (G_surf_file is not None) and (G_surf is None):
                np.save(G_surf_file, sim.G_surf)
                if verbose:
                    print(f"Saved G_surf to '{G_surf_file}'")
        # return new object
        return sim

    def write_run_files(self, folder):
        """
        Write all the data required to run the simulation into files and save type and
        shape information as a JSON. Output files will be overwritten.

        Parameters
        ----------
        folder : str
            Folder to use for the output files. Gets created if it doesn't exist.
        """
        # imports
        import json
        from pathlib import Path

        # check for folder, raises error if it's a file
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        # set up data
        data = {
            "t_eval_joint_sec": self.t_eval_joint * SEC_PER_YEAR,
            "ix_break_joint": self.ix_break_joint.astype("int32"),
            "ix_eq_joint": self.ix_eq_joint.astype("int32"),
            "K_inner_inner_onfault": self.fault.K_inner_inner[:, :2, :, :2],
            "K_inner_asperities_v_plate": self.K_inner_asperities_v_plate,
            "v_plate_ddcs_proj_eff_inner": self.v_plate_ddcs_proj_eff[
                self.fault.s_inner, :
            ],
            "v_init": self.v_init,
            "delta_tau_bounded_compressed": self.delta_tau_bounded_compressed,
            "delta_tau_bounded_indices": self.delta_tau_bounded_indices.astype("int32"),
            "v_0": np.atleast_1d(self.rheo.v_0),
            "alpha_h_vec": np.squeeze(self.alpha_h_vec),
            "mu_over_2vs": np.atleast_1d(self.fault.mu_over_2vs),
        }
        # set up info dictionary
        info = {}
        # write data
        for key, value in data.items():
            # write data
            with open(folder / f"{key}.bin", mode="wb") as fdata:
                np.ascontiguousarray(value).tofile(fdata)
            # attach info
            info[key] = {"dtype": str(value.dtype), "shape": value.shape}
        # write info
        with open(folder / "runfiles.json", mode="wt") as finfo:
            json.dump(info, finfo, indent=2)

    def run(self, reference_velocity_index=None, verbose=False, rho=None):
        """
        Run a full simulation.
        """
        # run forward integration
        if isinstance(self.rheo, RateStateSteadyLogarithmic) or isinstance(
            self.rheo, RateStateSteadyLogarithmic2D
        ):
            if rho is None:  # use default method
                sim_state = flat_run_rdlog(
                    self.t_eval_joint * SEC_PER_YEAR,
                    self.ix_break_joint,
                    self.ix_eq_joint,
                    self.fault.K_inner_inner[:, :2, :, :2],
                    self.K_inner_asperities_v_plate,
                    self.v_plate_ddcs_proj_eff[self.fault.s_inner, :],
                    self.v_init,
                    self.delta_tau_bounded,
                    self.delta_tau_bounded_nonuni,
                    self.rheo.v_0,
                    self.alpha_h_vec,
                    self.fault.mu_over_2vs,
                    self.v_max,
                    self.atol,
                    self.rtol,
                    self.spinup_atol,
                    self.spinup_rtol,
                    verbose,
                )
            else:  # use regularized version
                sim_state = flat_run_rdreg(
                    self.t_eval_joint * SEC_PER_YEAR,
                    self.ix_break_joint,
                    self.ix_eq_joint,
                    self.fault.K_inner_inner[:, :2, :, :2],
                    self.K_inner_asperities_v_plate,
                    self.v_plate_ddcs_proj_eff[self.fault.s_inner, :],
                    self.v_init,
                    self.delta_tau_bounded,
                    self.delta_tau_bounded_nonuni,
                    self.slip_taper_vec,
                    self.slip_taper_vec_nonuni,
                    rho,
                    self.alpha_h_vec,
                    self.fault.mu_over_2vs,
                    self.v_max,
                    self.atol,
                    self.rtol,
                    self.spinup_atol,
                    self.spinup_rtol,
                    verbose,
                )
        else:
            raise NotImplementedError
        # extract the observations that were actually requested
        obs_state = sim_state[:, :, self.ix_obs_joint]
        # convert to surface displacements
        surf_disps_sim = get_surface_displacements(
            obs_state[: self.fault.inner_num_patches, :, :],
            self.G_surf[:, :, self.fault.s_inner, :],
        )
        surf_disps_locked = get_surface_displacements(
            self.locked_slip, self.G_surf[:, :, self.fault.s_asperities, :]
        )
        surf_disps_outer = get_surface_displacements(
            self.outer_creep_slip, self.G_surf[:, :, self.fault.s_outer, :]
        )
        surf_disps_lower = get_surface_displacements(
            self.lower_creep_slip, self.G_surf[:, :, self.fault.s_lower, :]
        )
        # remove effect of reference velocity on surf_disps_sim if desired
        if reference_velocity_index is not None:
            ref_patch_vel = obs_state[
                self.fault.inner_num_patches :, :, reference_velocity_index
            ]
            ref_surf_vel = get_surface_displacements(
                ref_patch_vel[:, :, None], self.G_surf[:, :, self.fault.s_inner, :]
            )
            ref_surf_vel_ts = ref_surf_vel * (
                (self.t_obs - self.t_obs[0]) * SEC_PER_YEAR
            )
            surf_disps_sim -= ref_surf_vel_ts
        return (
            sim_state,
            obs_state,
            surf_disps_sim,
            surf_disps_locked,
            surf_disps_outer,
            surf_disps_lower,
        )

    def zero_obs_at_eq(self, surf_disps):
        r"""
        Reset to zero the surface displacement timeseries every time an earthquake happens
        and at the beginning (taking into account NaN values).

        Parameters
        ----------
        surf_disps : numpy.ndarray
            Surface displacements of shape :math:`(n_\text{stations}, 3, n_\text{observations})`
        """
        obs_zeroed = surf_disps.copy()
        obs_mask = np.isfinite(obs_zeroed)
        if not np.all(obs_mask):
            # need to loop over every station to find first valid observation after event
            periods_from = [0] + self.i_slips_obs
            periods_to = self.i_slips_obs + [obs_zeroed.shape[2]]
            for i_stat in range(obs_zeroed.shape[0]):
                for i_comp in range(obs_zeroed.shape[1]):
                    # iterate over all periods between bounds or earthquakes
                    for i_from, i_to in zip(periods_from, periods_to):
                        offset = None
                        for j in range(i_from, i_to):
                            # skip if data is masked
                            if obs_mask[i_stat, i_comp, j]:
                                if offset is None:
                                    # save new offset for this period
                                    offset = obs_zeroed[i_stat, i_comp, j]
                                # apply current offset
                                obs_zeroed[i_stat, i_comp, j] -= offset
        else:
            # no NaNs
            obs_zeroed -= obs_zeroed[:, :, 0][:, :, None]
            if self.i_slips_obs is not None:
                for i in range(self.n_slips_obs):
                    obs_zeroed[:, :, self.i_slips_obs[i] :] -= obs_zeroed[
                        :, :, self.i_slips_obs[i]
                    ][:, :, None]
        return obs_zeroed

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
        isort = np.argsort(self.pts_surf[:, 0])
        i_off = 1.5 * np.std(obs_zeroed, axis=(0, 2))
        # get float dates of observed earthquakes
        slips_obs = np.logical_and(
            self.t_obs.min() <= self.eq_df.index, self.t_obs.max() > self.eq_df.index
        )
        n_slips_obs = slips_obs.sum()
        if n_slips_obs > 0:
            i_slips_obs = [
                np.argmax(self.t_obs >= t_eq)
                for t_eq in self.eq_df.index.values[slips_obs]
            ]
            t_last_slips = [self.t_obs[islip] for islip in i_slips_obs]
        else:
            t_last_slips = []
        # start plot
        fig, ax = plt.subplots(nrows=3, sharex=True, layout="constrained")
        for tslip in t_last_slips:
            for a in ax:
                a.axvline(tslip, c="0.7", zorder=-1)
        for j, io in enumerate(i_off):
            for i, ix in enumerate(isort):
                if obs_noisy is not None:
                    ax[j].plot(
                        self.t_obs,
                        obs_noisy[ix, j, :] + i * io,
                        ".",
                        c="k",
                        rasterized=True,
                    )
                ax[j].plot(self.t_obs, obs_zeroed[ix, j, :] + i * io, c=f"C{i}")
            ax[j].set_ylabel(f"$x_{j + 1}$ [m]")
        ax[1].set_xlabel("Time")
        fig.suptitle("Surface Displacement")
        return fig, ax

    def plot_eq_stress(self, tdcs=True):
        """
        Plot the stress changes induced by the different earthquakes.

        Parameters
        ----------
        tdcs : bool, optional
            If ``True`` (default), plot in TDCS, otherwise in DDCS.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import SymLogNorm, Normalize
        from matplotlib.cm import ScalarMappable
        from cmcrameri import cm

        # get unique earthquakes
        (
            _,
            unique_indices,
        ) = np.unique(self.eq_df.values, True, axis=0)
        n_unique = unique_indices.size
        # rotate if necessary
        if tdcs:
            labels = ["Magnitude", "Strike", "Dip"]
            delta_tau_bounded = np.einsum(
                "ikj,lik->lij",
                self.fault.R_inner_tdcs_to_ddcs[:, :2, :2],
                self.delta_tau_bounded,
            )
            eq_proj = np.einsum(
                "ikj,lik->lij",
                self.fault.R_asperities_tdcs_to_ddcs[:, :2, :2],
                self.eq_ddcs_proj,
            )
        else:
            labels = ["Magnitude", "$R-45°$", "$R+45°$"]
            delta_tau_bounded = self.delta_tau_bounded
            eq_proj = self.eq_ddcs_proj
        # get slip data range and norm
        s_max = np.ceil(np.max(self.eq_df.max()))
        s_norm = Normalize(vmin=-s_max, vmax=s_max)
        eq_proj_mag = np.linalg.norm(eq_proj, axis=2)
        # start figures
        figs = [plt.figure() for _ in range(n_unique)]
        axes = []
        for i, fig in enumerate(figs):
            gs = gridspec.GridSpec(2, 20, figure=fig)
            ax_mag = fig.add_subplot(gs[0, :10], projection="3d")
            ax_cbar_tau = fig.add_subplot(gs[0, 12])
            ax_cbar_slip = fig.add_subplot(gs[0, 17])
            ax_dip = fig.add_subplot(gs[1, :10], projection="3d")
            ax_strike = fig.add_subplot(gs[1, 10:], projection="3d")
            axes3d = [ax_mag, ax_dip, ax_strike]
            axes.append(axes3d)
            colls_inner = [None for _ in range(3)]
            colls_asp = [None for _ in range(3)]
            for j, ax in enumerate(axes3d):
                # plot inner mesh
                colls_inner[j] = ax.plot_trisurf(
                    self.fault.inner_vertices[:, 0] / 1e3,
                    self.fault.inner_vertices[:, 1] / 1e3,
                    self.fault.inner_vertices[:, 2] / 1e3,
                    triangles=self.fault.inner_triangles,
                    antialiased=True,
                    linewidths=0,
                )
                # plot asperities
                colls_asp[j] = ax.plot_trisurf(
                    self.fault.asperities_vertices[:, 0] / 1e3,
                    self.fault.asperities_vertices[:, 1] / 1e3,
                    self.fault.asperities_vertices[:, 2] / 1e3,
                    triangles=self.fault.asperities_triangles,
                    antialiased=True,
                    linewidths=0,
                )
                # make prettier
                ax.view_init(elev=20, azim=-135)
                ax.set_title(labels[j])
                ax.set_xlabel("E [km]")
                ax.set_ylabel("N [km]")
                ax.set_zlabel("U [km]")
                ax.set_box_aspect(
                    [
                        ub - lb
                        for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")
                    ]
                )
            # create data norms for delta_tau
            dtau = delta_tau_bounded[unique_indices[i], :, :]
            dtau_mag = np.linalg.norm(dtau, axis=1)
            dtau_mag_oom = 10 ** np.ceil(np.log10(np.max(dtau_mag)))
            dtau_norm = SymLogNorm(
                vmin=-dtau_mag_oom, vmax=dtau_mag_oom, linthresh=dtau_mag_oom / 1e3
            )
            # color inner mesh according to stress change in the different components, and
            # color asperities according to slip
            # top left: magnitude
            colls_inner[0].set_facecolor([cm.vik(dtau_norm(q)) for q in dtau_mag])
            colls_asp[0].set_facecolor(
                [cm.bam(s_norm(q)) for q in eq_proj_mag[unique_indices[i], :]]
            )
            # bottom left: strike
            colls_inner[1].set_facecolor([cm.vik(dtau_norm(q)) for q in dtau[:, 0]])
            colls_asp[1].set_facecolor(
                [cm.bam(s_norm(q)) for q in eq_proj[unique_indices[i], :, 0]]
            )
            # bottom right: dip
            colls_inner[2].set_facecolor([cm.vik(dtau_norm(q)) for q in dtau[:, 1]])
            colls_asp[2].set_facecolor(
                [cm.bam(s_norm(q)) for q in eq_proj[unique_indices[i], :, 1]]
            )
            # add colorbars
            fig.colorbar(
                ScalarMappable(norm=dtau_norm, cmap=cm.vik),
                cax=ax_cbar_tau,
                label=r"$\Delta \tau$ [Pa]",
            )
            fig.colorbar(
                ScalarMappable(norm=s_norm, cmap=cm.bam),
                cax=ax_cbar_slip,
                label=r"$\Delta s$ [m]",
            )
        # finish
        return figs, axes

    def plot_field(self, quantity, log=True, label=None):
        """
        Plot a quantity on top of the fault mesh.

        Parameters
        ----------
        quantity : numpy.ndarray
            Field quantity of the same shape as the inner mesh.
        log : bool, optional
            If ``True`` (default), use a logarithmic colorscale.
        label : str, optional
            Label of the colorbar.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm, LogNorm, Normalize
        from matplotlib.cm import ScalarMappable
        from cmcrameri import cm

        # check data input size
        assert isinstance(quantity, np.ndarray)
        assert quantity.size == self.fault.inner_num_patches, (
            f"Expected input quantity to be of size {self.fault.inner_num_patches}, "
            f"got {quantity.shape}"
        )
        # get data range and norm
        if log:
            q_oom = 10 ** np.ceil(np.log10(np.max(np.abs(quantity))))
            if np.any(quantity < 0):
                q_norm = SymLogNorm(vmin=-q_oom, vmax=q_oom, linthresh=q_oom / 1e5)
                q_cmap = cm.vik
            else:
                q_norm = LogNorm(
                    vmin=10 ** np.floor(np.log10(np.min(quantity))), vmax=q_oom
                )
                q_cmap = cm.batlow
        else:
            q_absmax = np.max(np.abs(quantity))
            if np.any(quantity < 0):
                q_norm = Normalize(vmin=-q_absmax, vmax=q_absmax)
                q_cmap = cm.vik
            else:
                q_norm = Normalize(vmin=0, vmax=q_absmax)
                q_cmap = cm.batlow
        # start figure
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # plot inner mesh
        coll = ax.plot_trisurf(
            self.fault.inner_vertices[:, 0] / 1e3,
            self.fault.inner_vertices[:, 1] / 1e3,
            self.fault.inner_vertices[:, 2] / 1e3,
            triangles=self.fault.inner_triangles,
            antialiased=True,
            linewidths=0,
        )
        # make prettier
        ax.view_init(elev=20, azim=-135)
        ax.set_xlabel("E [km]")
        ax.set_ylabel("N [km]")
        ax.set_zlabel("U [km]")
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        # set mesh colors
        coll.set_facecolor([q_cmap(q_norm(q)) for q in quantity])
        # add colorbar
        fig.colorbar(ScalarMappable(norm=q_norm, cmap=q_cmap), ax=ax, label=label)
        # finish
        return fig, ax


@dataclass
class SEASResult:
    """Class that contains self-consistent fault, surface, and rheological data"""

    slip: np.ndarray | None = None
    sliprate: np.ndarray | None = None
    disp: np.ndarray | None = None
    vel: np.ndarray | None = None
    v_plate: np.ndarray | None = None
    alpha_h: np.ndarray | None = None
    t_eff: float | None = None
    n_samples: int | None = field(init=False)
    n_patches: int | None = field(init=False)
    n_stations: int | None = field(init=False)
    n_observations: int | None = field(init=False)
    _slipdeficit: np.ndarray | None = field(default=None, init=False)
    _coupling: np.ndarray | None = field(default=None, init=False)
    _global_mask: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        """Perform checks and initialize sizes"""
        n_dims = [
            arr.ndim
            for arr in [self.slip, self.sliprate, self.disp, self.vel]
            if arr is not None
        ]
        if len(n_dims) > 0:
            assert len(set(n_dims)) == 1
            n_dims = n_dims[0]
            if n_dims == 3:
                dim_off = 0
                self.n_samples = 1
            elif n_dims == 4:
                dim_off = 1
                n_samples = [
                    arr.shape[0]
                    for arr in [self.slip, self.sliprate, self.disp, self.vel]
                    if arr is not None
                ]
                assert len(set(n_samples)) == 1
                self.n_samples = n_samples[0]
            else:
                raise ValueError
            n_observations = [
                arr.shape[dim_off + 2]
                for arr in [self.slip, self.sliprate, self.disp, self.vel]
                if arr is not None
            ]
            assert len(set(n_observations)) == 1
            self.n_observations = n_observations[0]
            n_stations = [
                arr.shape[dim_off + 0]
                for arr in [self.disp, self.vel]
                if arr is not None
            ]
            assert len(set(n_stations)) == 1
            self.n_stations = n_stations[0]
            n_patches = [
                arr.shape[dim_off + 0]
                for arr in [self.slip, self.sliprate]
                if arr is not None
            ]
            assert len(set(n_patches)) == 1
            self.n_patches = n_patches[0]
        else:
            n_dims = None  # no time-variable arrays
            self.n_samples = None
            self.n_observations = None
            self.n_stations = None
            self.n_patches = None
        if self.v_plate is not None:
            assert self.v_plate.shape[1] == 2
            if self.n_patches is None:
                self.n_patches = self.v_plate.shape[0]
            else:
                assert self.n_patches == self.v_plate.shape[0]
        if self.alpha_h is not None:
            if self.alpha_h.ndim == 1:
                dim_off = 0
                if self.n_samples is None:
                    self.n_samples = 1
                else:
                    assert self.n_samples == 1
            elif self.alpha_h.ndim == 2:
                dim_off = 1
                if self.n_samples is None:
                    self.n_samples = self.alpha_h.shape[0]
                else:
                    assert self.n_samples == self.alpha_h.shape[0]
            else:
                raise ValueError
            if self.n_patches is None:
                self.n_patches = self.alpha_h.shape[dim_off + 0]
            else:
                assert self.n_patches == self.alpha_h.shape[dim_off + 0]

    def __repr__(self):
        """String representation of object"""
        dims_str = ", ".join(
            [
                f"{n}={getattr(self, n)}"
                for n in [
                    "n_samples",
                    "n_observations",
                    "n_stations",
                    "n_patches",
                    "t_eff",
                ]
            ]
        )
        repr_str = f"SEASResult({dims_str}):\n"
        repr_str += "\n".join(
            [
                f"  {n}: {'missing' if getattr(self, n) is None
                       else str(getattr(self, n).shape)}"
                for n in ["slip", "sliprate", "disp", "vel", "v_plate", "alpha_h"]
            ]
        )
        return repr_str

    def get_slipdeficit(self):
        """
        Calculate the slip deficit on the fault.

        Returns
        -------
        numpy.ndarray
            Normalized slip dificit with shape `(n_samples, n_patches, n_observations).
        """
        if (self.slip is None) or (self.v_plate is None) or (self.t_eff is None):
            raise ValueError(
                "Need 'slip', 'v_plate', and 't_eff' to be set "
                "to calculate slip deficit."
            )
        else:
            v_plate_mag = np.linalg.norm(self.v_plate, axis=1)
            slip_vp = np.linalg.norm(self.slip, axis=-2) * np.cos(
                np.pi / 4 - np.arctan2(self.slip[..., 1, :], self.slip[..., 0, :])
            )
            if self.disp.ndim == 3:
                return 1 - slip_vp / (v_plate_mag[:, None] * SEC_PER_YEAR * self.t_eff)
            else:  # self.disp.ndim == 4:
                return 1 - slip_vp / (
                    v_plate_mag[None, :, None] * SEC_PER_YEAR * self.t_eff
                )

    @property
    def slipdeficit(self):
        # need to calculate if it's the first time
        if self._slipdeficit is None:
            self._slipdeficit = self.get_slipdeficit()
        # return calculated value
        return self._slipdeficit

    def get_coupling(self):
        """
        Calculate the kinematic coupling on the fault.

        Returns
        -------
        numpy.ndarray
            Normalized kinematic coupling, same shape as `slip`.
        """
        if (self.sliprate is None) or (self.v_plate is None):
            raise ValueError(
                "Need 'sliprate' and 'v_plate' to be set "
                "to calculate kinematic coupling."
            )
        else:
            v_plate_mag = np.linalg.norm(self.v_plate, axis=1)
            sliprate_vp = np.linalg.norm(self.sliprate, axis=-2) * np.cos(
                np.pi / 4
                - np.arctan2(self.sliprate[..., 1, :], self.sliprate[..., 0, :])
            )
            if self.disp.ndim == 3:
                return 1 - sliprate_vp / v_plate_mag[:, None]
            else:  # self.disp.ndim == 4:
                return 1 - sliprate_vp / v_plate_mag[None, :, None]

    @property
    def coupling(self):
        # need to calculate if it's the first time
        if self._coupling is None:
            self._coupling = self.get_coupling()
        # return calculated value
        return self._coupling

    def get_global_mask(self):
        """
        Calculate indices where all displacements are missing.

        Returns
        -------
        numpy.ndarray
            List of indices [-].
        """
        if self.disp is None:
            raise ValueError("Need 'disp' to be set to calculate mask.")
        else:
            if self.disp.ndim == 3:
                return np.flatnonzero(np.all(np.isnan(self.disp), axis=(0, 1)))
            else:  # self.disp.ndim == 4:
                return np.flatnonzero(np.all(np.isnan(self.disp[0, ...]), axis=(0, 1)))

    @property
    def global_mask(self):
        # need to calculate if it's the first time
        if self._global_mask is None:
            self._global_mask = self.get_global_mask()
        # return calculated value
        return self._global_mask

    @classmethod
    def from_sim_run_output(
        cls, sim, sim_state, surf_disps_sim, surf_disps_outer, surf_disps_lower
    ):
        """
        Create a SEASResult object from the the output of `SubductionSimulation3D.run`.

        Parameters
        ----------
        sim : SubductionSimulation3D
            Simulation object
        obs_state : numpy.ndarray
            `obs_state` from `SubductionSimulation3D.run`
        surf_disps_sim : numpy.ndarray
            `surf_disps_sim` from `SubductionSimulation3D.run`
        surf_disps_outer : numpy.ndarray
            `surf_disps_outer` from `SubductionSimulation3D.run`
        surf_disps_lower : numpy.ndarray
            `surf_disps_lower` from `SubductionSimulation3D.run`

        Returns
        -------
        SEASResult
        """
        slip = (
            sim_state[: sim.fault.inner_num_patches, :, sim.ix_obs_joint]
            - sim_state[: sim.fault.inner_num_patches, :, sim.ix_break_joint[-2]][
                :, :, None
            ]
            + sim.slip_taper
        )
        sliprate = sim_state[sim.fault.inner_num_patches :, :, sim.ix_obs_joint]
        disp = sim.zero_obs_at_eq(surf_disps_sim + surf_disps_outer + surf_disps_lower)
        vel = sum(sim.get_vels(sim_state[:, :, sim.ix_obs_joint]))
        v_plate = sim.v_plate_ddcs_proj_eff[sim.fault.s_inner, :]
        alpha_h = sim.alpha_h_vec
        return cls(slip, sliprate, disp, vel, v_plate, alpha_h)

    @staticmethod
    def load(path):
        """Load a pickled instance of this object"""
        with open(path, mode="rb") as f:
            return pickle.load(f)

    def save(self, path):
        """Save instance to a pickled file"""
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
