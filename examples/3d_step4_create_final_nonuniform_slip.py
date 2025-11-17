"""
Commands used to create the final non-uniform slip distribution.
"""

# imports
import json
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from pathlib import Path
from load_srcmod_mat import (
    load_file as load_srcmod_file,
    extract_slip as extract_srcmod_slip,
)
from seqeas.subduction3d import read_tet
from multiprocessing import Pool

# define paths
ROOT_DIR = None  # path as a Path object to the downloaded slip models
MESHFOLDER = None  # path to the folder with the mesh
assert ROOT_DIR is not None, "'ROOT_DIR' must be set"
assert MESHFOLDER is not None, "'MESHFOLDER' must be set"
# constants
PROJ_GEOD = ccrs.Geodetic()
PROJ_UTM = ccrs.UTM(zone=54)
POINTWISE_DATA_TYPES = ["srcmod", "usgs", "grid"]
ASP_MESH_FILES = [f"{MESHFOLDER}/mesh_elli{i}.tet" for i in range(1, 8)]
FINAL_NONUNI_FILENAME = f"{MESHFOLDER}/final_nonuniform_slip.npy"
NUM_THREADS = 8
T_LAST = [
    "2011-03-11 05:46:24",
    "2011-03-11 05:46:24",
    "1994-12-28 12:19:23",
    "2003-09-25 19:50:06",
    "2003-09-25 19:50:06",
    "2011-03-11 05:46:24",
    "1653-09-25 19:50:06",
]
T_REC = [80, 40, 40, 50, 50, 1000, 500]  # [a]
T_REC_LOGSIGMA = 0
V_P = 0.095  # [mm/a]
EQ_SETUP_FILENAME = f"{MESHFOLDER}/eq_setup.csv"
D_0_LOGSIGMA = 0
M_0 = [4.335e20, 2.212e21, 5.593e22]  # [Nm] from USGS catalog
I_ASP_HONSHU = [0, 1, 2, 5]
MU = 45e9


# multiprocessing helper function
def par_point_in_patch(points_and_geoms: tuple) -> np.ndarray:
    """
    Check if a point is inside a collection of patches in parallel.

    Parameters
    ----------
    points_and_geoms
        Tuple of points and patch geometries

    Returns
    -------
        Boolean mask
    """
    patch, geoms = points_and_geoms
    return np.flatnonzero([patch.contains(p) for p in geoms])


# area of triangle in 3D
def tri_area_3d(vertices: np.ndarray) -> float:
    """
    Calculate the area of a triangle defined in 3D.

    Parameters
    ----------
    vertices
        Array with 3D coordinates for each point in each row

    Returns
    -------
        Area
    """
    ab = vertices[1, :] - vertices[0, :]
    ac = vertices[2, :] - vertices[0, :]
    return np.linalg.norm(np.cross(ab, ac)) / 2


if __name__ == "__main__":

    # import all data
    raw = {}
    for asp_dir in sorted(ROOT_DIR.iterdir()):
        if str(asp_dir).endswith(".backup"):
            continue
        asp_dict = {}
        for year_dir in sorted(asp_dir.iterdir()):
            year_dict = {}
            # check for SRCMOD files with .mat extension
            for mat_file in sorted(year_dir.glob("*.mat")):
                # load file
                srcmod_file = load_srcmod_file(mat_file)
                mdl_segments = extract_srcmod_slip(srcmod_file)
                # concatenate all segment slip centroids
                mdl_slip = np.concatenate(
                    [
                        np.stack(
                            [
                                getattr(s, attr).ravel()
                                for attr in ["lon", "lat", "depth", "slip"]
                            ],
                            axis=1,
                        )
                        for s in mdl_segments
                    ],
                    axis=0,
                )
                # convert depth to meters
                mdl_slip[:, 2] *= 1000
                # save to dictionary
                year_dict[mat_file.stem] = ("srcmod", mdl_slip[:, 3].max(), mdl_slip)
            # check for .geojson files which can either be USGS or self-digitized models
            for geojson_file in sorted(year_dir.glob("*.geojson")):
                # load file
                with open(geojson_file, mode="rt") as f:
                    json_file = json.load(f)
                # check if there are other files with the same basename
                if len(list(year_dir.glob(f"{geojson_file.stem}*"))) == 1:
                    # it's a USGS finite fault model
                    mdl_slip = np.stack(
                        [
                            np.array(feat["geometry"]["coordinates"][0])
                            .mean(axis=0)
                            .tolist()
                            + [feat["properties"]["slip"]]
                            for feat in json_file["features"]
                        ],
                        axis=0,
                    )
                    # save to dictionary
                    year_dict[geojson_file.stem] = (
                        "usgs",
                        mdl_slip[:, 3].max(),
                        mdl_slip,
                    )
                else:
                    # it's a self-digitized shape
                    mdl_segments = [
                        np.array(feat["geometry"]["coordinates"][0])
                        for feat in json_file["features"]
                    ]
                    npy_files = list(year_dir.glob(f"{geojson_file.stem}*.npy"))
                    csv_files = list(year_dir.glob(f"{geojson_file.stem}*.csv"))
                    if len(npy_files) == 1:  # we have gridded slip data
                        assert len(mdl_segments) == 1
                        mdl_border = mdl_segments[0]
                        # assume the polygon starts bottom right and then goes counter-
                        # clockwise, we need the grid points based on the NumPy array shape
                        mdl_array = np.load(npy_files[0])
                        mdl_nx, mdl_ny = mdl_array.shape
                        mdl_vec_down = mdl_border[3, :] - mdl_border[2, :]
                        mdl_vec_right = mdl_border[1, :] - mdl_border[2, :]
                        mdl_lons = np.empty_like(mdl_array)
                        mdl_lats = np.empty_like(mdl_array)
                        for i in range(mdl_ny):
                            for j in range(mdl_nx):
                                mdl_lons[i, j], mdl_lats[i, j] = (
                                    mdl_border[2, :]
                                    + (0.5 + i) / mdl_ny * mdl_vec_down
                                    + (0.5 + j) / mdl_nx * mdl_vec_right
                                )
                        # stack and set depth = 0
                        mdl_slip = np.stack(
                            [
                                mdl_lons.ravel(),
                                mdl_lats.ravel(),
                                np.full(mdl_array.size, 0),
                                mdl_array.ravel(),
                            ],
                            axis=1,
                        )
                        # save to dictionary
                        year_dict[geojson_file.stem] = (
                            "grid",
                            mdl_slip[:, 3].max(),
                            mdl_slip,
                        )
                    elif len(csv_files) == 1:  # we have contour data
                        mdl_slip = pd.read_csv(csv_files[0], comment="#")[
                            "Dislocation (m)"
                        ].values
                        assert mdl_slip.size == len(mdl_segments)
                        year_dict[geojson_file.stem] = (
                            "contours",
                            mdl_slip.max(),
                            (mdl_segments, mdl_slip),
                        )
                    else:  # unknown format
                        raise RuntimeError(
                            "Multiple NumPy and/or CSV files found: "
                            f"{npy_files + csv_files}."
                        )
            # save to year dictionary
            asp_dict[year_dir.stem] = year_dict
        # save to root dictionary
        raw[asp_dir.stem] = asp_dict

    # load asperity meshes
    asp_verts, asp_tris, asp_polys = [], [], []
    for mf in ASP_MESH_FILES:
        temp_verts, temp_tris = read_tet(mf)
        temp_xy = [
            np.array(
                sgeom.Polygon(temp_verts[temp_tris[i, :]]).buffer(10e3).exterior.coords
            )
            for i in range(temp_tris.shape[0])
        ]
        temp_lla = [
            PROJ_GEOD.transform_points(PROJ_UTM, txy[:, 0], txy[:, 1])
            for txy in temp_xy
        ]
        temp_poly = [sgeom.Polygon(tlla[:, :2]) for tlla in temp_lla]
        asp_verts.append(temp_verts)
        asp_tris.append(temp_tris)
        asp_polys.append(temp_poly)
    asp_centrs = [
        np.concatenate([p.centroid.coords for p in po], axis=0) for po in asp_polys
    ]
    asp_areas = [
        [tri_area_3d(av[at][i]) for i in range(len(at))]
        for av, at in zip(asp_verts, asp_tris)
    ]
    asp_areas_flat = np.array([aa for a in asp_areas for aa in a])

    # combine all pointwise data based on asperity patches
    asp_agg = {
        asp: {year: None for year in asp_dict.keys()} for asp, asp_dict in raw.items()
    }
    ix_point_in_asp = {
        asp: {year: None for year in asp_dict.keys()} for asp, asp_dict in raw.items()
    }
    with Pool(NUM_THREADS) as p:
        for iasp, (asp, asp_dict) in enumerate(raw.items()):
            if list(asp_dict.keys())[-1] >= "1990":
                year, year_dict = list(asp_dict.items())[-1]
                print(asp, year)
                if any([mdl[0] in POINTWISE_DATA_TYPES for mdl in year_dict.values()]):
                    points = np.concatenate(
                        [
                            mdl[2][:, :2]
                            for mdl in year_dict.values()
                            if mdl[0] in POINTWISE_DATA_TYPES
                        ],
                        axis=0,
                    )
                    values = np.concatenate(
                        [
                            mdl[2][:, 3]
                            for mdl in year_dict.values()
                            if mdl[0] in POINTWISE_DATA_TYPES
                        ],
                        axis=0,
                    )
                    points_coll = sgeom.MultiPoint(points)
                    iterables = (
                        (patch, points_coll.geoms) for patch in asp_polys[iasp]
                    )
                    ix_point_in_asp[asp][year] = list(
                        p.imap(par_point_in_patch, iterables)
                    )
                    asp_agg[asp][year] = np.array(
                        [
                            np.mean(values[pip]) if len(pip) > 0 else 0
                            for pip in ix_point_in_asp[asp][year]
                        ]
                    )
                elif any([mdl[0] == "contours" for mdl in year_dict.values()]):
                    conts = [m for mdl in year_dict.values() for m in mdl[2][0]]
                    cont_rings = [sgeom.LinearRing(c) for c in conts]
                    values = np.concatenate(
                        [mdl[2][1] for mdl in year_dict.values()], axis=0
                    )
                    temp = [
                        values[
                            [
                                sgeom.Polygon(cont).contains(sgeom.Point(cent))
                                for cont in cont_rings
                            ]
                        ]
                        for cent in asp_centrs[iasp]
                    ]
                    asp_agg[asp][year] = np.array(
                        [max(v) if len(v) > 0 else 0 for v in temp]
                    )
        # override miyagi with tohoku slip
        iasp, (asp, asp_dict), year, year_dict = (
            1,
            ("2_Miyagi", raw["2_Miyagi"]),
            "2011",
            raw["6_Tohoku"]["2011"],
        )
        points = np.concatenate(
            [
                mdl[2][:, :2]
                for mdl in year_dict.values()
                if mdl[0] in POINTWISE_DATA_TYPES
            ],
            axis=0,
        )
        values = np.concatenate(
            [
                mdl[2][:, 3]
                for mdl in year_dict.values()
                if mdl[0] in POINTWISE_DATA_TYPES
            ],
            axis=0,
        )
        points_coll = sgeom.MultiPoint(points)
        iterables = ((patch, points_coll.geoms) for patch in asp_polys[iasp])
        ix_point_in_asp[asp][year] = list(p.imap(par_point_in_patch, iterables))
        asp_agg[asp] = {
            year: np.array(
                [
                    np.mean(values[pip]) if len(pip) > 0 else 0
                    for pip in ix_point_in_asp[asp][year]
                ]
            )
        }
        # override nemuro 2004 as 0, is already close to that, also sync with 2003
        asp_agg["5_Nemuro"] = {"2003": 0 * asp_agg["5_Nemuro"]["2004"]}

    # create cyclic uniform slip
    D_0 = np.diag(V_P * np.array(T_REC))
    D_0_logsigma = D_0_LOGSIGMA * (D_0 > 0).astype(float)

    # create final non-uniform slip
    years = sorted(
        set([kk for v in asp_agg.values() for kk, vv in v.items() if vv is not None])
    )
    ranges_asp = np.cumsum([0] + [tri.shape[0] for tri in asp_tris])
    final_slip = np.zeros((len(years), sum([tri.shape[0] for tri in asp_tris])))
    for iasp, asp_dict in enumerate(asp_agg.values()):
        year, slip = list(asp_dict.items())[-1]
        print(year, slip)
        if slip is None:
            continue
        iy = years.index(year)
        print(iy)
        final_slip[iy, ranges_asp[iasp] : ranges_asp[iasp + 1]] = slip
    # rescale to match earthquake moment
    assert final_slip.shape == (len(M_0), asp_areas_flat.size)
    for i in range(final_slip.shape[0]):
        curr_moment = np.sum(final_slip[i, :] * asp_areas_flat) * MU
        factor = M_0[i] / curr_moment
        print(f"Rescaling slip for EQ {i + 1} by a factor of {factor}.")
        final_slip[i, :] *= factor

    # save
    print("Full final_slip array:")
    print(np.round(final_slip).astype(int))
    np.save(FINAL_NONUNI_FILENAME, final_slip)
    """
    [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  2,
       1,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  7,  9,  6,  5,
       7,  5,  5,  9,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [ 0,  9,  3,  3,  3,  3,  3,  6,  9,  3, 40, 41,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 65, 74, 59, 59,
      67, 65, 65, 72, 48, 69, 74, 61, 63, 56, 77, 66, 59, 67, 66, 75,
      56, 80, 72, 73, 76, 72,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
    """

    # create earthquake setup file
    eq_setup = pd.DataFrame(
        data={
            **{
                "T_last": T_LAST,
                "T_rec": T_REC,
                "T_rec_logsigma": [T_REC_LOGSIGMA] * len(T_REC),
            },
            **{f"D_0_{i}": D_0[:, i] for i in range(len(T_REC))},
            **{f"D_0_logsigma_{i}": D_0_logsigma[:, i] for i in range(len(T_REC))},
        }
    )
    eq_setup.index.names = ["EQ"]
    print(eq_setup)
    eq_setup.to_csv(EQ_SETUP_FILENAME)
    """
                    T_last T_rec T_rec_logsigma D_0_0 D_0_1 D_0_2 D_0_3 D_0_4 D_0_5 D_0_6
    EQ
    0  2011-03-11 05:46:24    80              0   7.6   0.0   0.0  0.00  0.00   0.0   0.0
    1  2011-03-11 05:46:24    40              0   0.0   3.8   0.0  0.00  0.00   0.0   0.0
    2  1994-12-28 12:19:23    40              0   0.0   0.0   3.8  0.00  0.00   0.0   0.0
    3  2003-09-25 19:50:06    50              0   0.0   0.0   0.0  4.75  0.00   0.0   0.0
    4  2003-09-25 19:50:06    50              0   0.0   0.0   0.0  0.00  4.75   0.0   0.0
    5  2011-03-11 05:46:24  1000              0   0.0   0.0   0.0  0.00  0.00  95.0   0.0
    6  1653-09-25 19:50:06   500              0   0.0   0.0   0.0  0.00  0.00   0.0  47.5
    """

    # create the same files but only for the Honshu asperities
    range_asp_hon = np.zeros(final_slip.shape[1], dtype=bool)
    for iasp in I_ASP_HONSHU:
        range_asp_hon[ranges_asp[iasp] : ranges_asp[iasp + 1]] = True
    final_slip_hon = final_slip[:, range_asp_hon]
    ieq_in_hon = np.any(final_slip_hon, axis=1)
    final_slip_hon = final_slip_hon[ieq_in_hon, :]
    eq_setup_hon = pd.DataFrame(
        data=eq_setup.iloc[
            I_ASP_HONSHU,
            np.r_[
                np.arange(3),
                np.array(I_ASP_HONSHU) + 3,
                np.array(I_ASP_HONSHU) + 3 + len(T_REC),
            ],
        ].values,
        columns=(
            ["T_last", "T_rec", "T_rec_logsigma"]
            + [f"D_0_{i}" for i in range(len(I_ASP_HONSHU))]
            + [f"D_0_logsigma_{i}" for i in range(len(I_ASP_HONSHU))]
        ),
    )
    eq_setup_hon.index.names = ["EQ"]
    np.save(FINAL_NONUNI_FILENAME.split(".")[-2] + "_hon.npy", final_slip_hon)
    eq_setup_hon.to_csv(EQ_SETUP_FILENAME.split(".")[-2] + "_hon.csv")

    # calculate slip variations to test influence on inversion results

    # variation 1: taper to zero inside the asperity
    # get Tohoku centroid
    toh_asp_centr = np.mean(asp_verts[5][asp_tris[5]], axis=(0, 1))
    # get distances from centroid
    toh_asp_dists = np.array(
        [
            np.linalg.norm(
                toh_asp_centr - np.mean(asp_verts[5][asp_tris[5][i]], axis=0)
            )
            for i in range(len(asp_tris[5]))
        ]
    )
    # calculate new slip ratios
    L = 100e3
    toh_asp_ratios_var1 = np.cos(toh_asp_dists / L / 2 * np.pi) ** 2
    # scale up to original share of moment
    scale_var1 = np.sum(
        final_slip[-1, ranges_asp[5] : ranges_asp[6]]
        * asp_areas_flat[ranges_asp[5] : ranges_asp[6]]
    ) / np.sum(toh_asp_ratios_var1 * asp_areas_flat[ranges_asp[5] : ranges_asp[6]])
    toh_asp_slip_var1 = toh_asp_ratios_var1 * scale_var1

    # variation 2: ramp the slip from zero to max at the trench
    # get Tohoku depths
    toh_asp_depths = np.mean(asp_verts[5][asp_tris[5]], axis=1)[:, 2]
    # calculate new slip ratios
    toh_asp_ratios_var2 = (toh_asp_depths - toh_asp_depths.min()) / (
        toh_asp_depths.max() - toh_asp_depths.min()
    )
    # scale up to original share of moment
    scale_var2 = np.sum(
        final_slip[-1, ranges_asp[5] : ranges_asp[6]]
        * asp_areas_flat[ranges_asp[5] : ranges_asp[6]]
    ) / np.sum(toh_asp_ratios_var2 * asp_areas_flat[ranges_asp[5] : ranges_asp[6]])
    toh_asp_slip_var2 = toh_asp_ratios_var2 * scale_var2

    # save variations for both global & Honshu cases
    final_slip_var1 = final_slip.copy()
    final_slip_var1[-1, ranges_asp[5] : ranges_asp[6]] = toh_asp_slip_var1
    final_slip_var2 = final_slip.copy()
    final_slip_var2[-1, ranges_asp[5] : ranges_asp[6]] = toh_asp_slip_var2
    final_slip_hon_var1 = final_slip_var1[np.ix_(ieq_in_hon, range_asp_hon)]
    final_slip_hon_var2 = final_slip_var2[np.ix_(ieq_in_hon, range_asp_hon)]
    np.save(FINAL_NONUNI_FILENAME.split(".")[-2] + "_var1.npy", final_slip_var1)
    np.save(FINAL_NONUNI_FILENAME.split(".")[-2] + "_var2.npy", final_slip_var2)
    np.save(FINAL_NONUNI_FILENAME.split(".")[-2] + "_hon_var1.npy", final_slip_hon_var1)
    np.save(FINAL_NONUNI_FILENAME.split(".")[-2] + "_hon_var2.npy", final_slip_hon_var2)
