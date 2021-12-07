import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import math
import os
import subprocess
from sys import platform
from plyfile import PlyData, PlyElement
from matplotlib.font_manager import FontProperties
import pandas as pd

def plot_scatter_pcs(pcs):
    '''
    initializes a figure with n voxelized pointclouds as scatter plots
    Parameters:
        pcs (np.ndarray): list of arrays of shape (H, W, D, c) where c 
                          is the number of 
                          channels, c can be 1 (onyl geometry) or 4 (geometry)
                          + color and the color should be in the 0-1 range
    Returns:
        axs (list): list of axs so they can be customized outside
    '''
    
    axs = []
    fig = plt.figure()
    for i, pc in enumerate(pcs):

        #checking that the input is correct
        if len(pc.shape) != 4:
            plt.close(fig)
            raise Exception("The input shape should be of type (H, W, D, c)")
        if pc.shape[3] != 1 and pc.shape[3] != 4:
            plt.close(fig)
            raise Exception("The number of channels should be 1 or 4")
        if np.max(pc > 1) or np.min(pc) < 0:
            plt.close(fig)
            raise Exception("The values for color should be in the [0, 1] range")
        
        
        ax = fig.add_subplot(int(f"1{len(pcs)}{i + 1}"), projection="3d")
        coords = np.where(pc[:, :, :, 0] > 0)
        if pc[coords].shape[1] > 1:
            colors = pc[coords][:, 1:]
        else: 
            colors = 0.5 * np.ones((pc[coords].shape[0], 3))
        ax.scatter(*coords, c=colors)
        axs.append(ax)

    return axs




def flatten_cubes(vol, nvx, nb):
    '''
    flattens subcubes in cube of size nb*nb*nb i.e. values from the same
    subcube are disposed one after the other
    Parameters:
        vol (np.ndarray): volume to flatten of size (nvx, nvx, nvx, c) where
                          c is the number of channels
        nvx (int): cube size along one dimension
        nb (int): subcube size along one dimension
    Return
        vol (np.ndarray): flattened volume of size 
                          (nvx/nb, nvx/nb, nvx/nb, nb**3, c)
    '''
    nbl = nvx // nb
    # divides each of the dimensions in nbl parts of dimension nb
    vol = vol.reshape(nbl, nb, nbl, nb, nbl, nb, -1)
    # places all nearby the axis relative to the nbs and in the correct order
    # so reshaping actually yields the correct vector
    vol = vol.swapaxes(1, 4).swapaxes(3, 4).swapaxes(1, 2).reshape(
        nbl, nbl, nbl, nb ** 3, -1
    )
    return vol

# reshapes vol so it goes back to how it was before flatten_cubes


def unflatten_cubes(vol, nvx, nb):
    '''
    Inverse operation of flatten cubes
    Parameters:
        vol (np.ndarray): volume to unflatten of size 
                          (nvx/nb, nvx/nb, nvx/nb, nb**3, c) where
                          c is the number of channels
        nvx (int): cube size along one dimension
        nb (int): subcube size along one dimension
    Return
        vol (np.ndarray): reshaped volume of size
                          (nvx, nvx, nvx, c)
    '''
    nbl = nvx // nb
    vol = vol.reshape(nbl, nbl, nbl, nb, nb, nb, -1)
    vol = vol.swapaxes(1, 2).swapaxes(3, 4).swapaxes(1, 4).reshape(
                nvx, nvx, nvx, -1
    )
    return vol

def voxelize_PC(pc, n_voxels=1024):
    '''
        voxelizes the point cloud in a cube with n_voxels voxels per dimension

        Parameters:
            pc (np.ndarray): the point cloud to be voxelized with
                            shape (Npts, nattr)
            n_voxels (int): number of voxels per dimension
        Returns:
            pc (np.ndarray): voxelized version of pc

    '''

    # finding the maximum delta of the PC
    min_val = np.min(pc[:, :3], axis=0)
    max_val = np.max(pc[:, :3], axis=0)
    dim_range = (np.max(max_val - min_val) + 0.0001) / n_voxels
    delta_m = min_val + (max_val - min_val) / 2

    # normalizing the PC to have a 10 bit representation
    pc[:, :3] = pc[:, :3] - np.reshape(delta_m, (1, -1))
    pc[:, :3] = np.floor(pc[:, :3] / dim_range) + n_voxels//2

    return pc

def plot_pointcloud(ptc: np.ndarray, name, color=None, colorset=None) -> None:
    """
    Plot a point cloud
        Parameters:
            ptc: point cloud to plot
    """

    label_to_names = ['unlabeled',
                      'car',
                      'bicycle',
                      'motorcycle',
                      'truck',
                      'other-vehicle',
                      'person',
                      'bicyclist',
                      'motorcyclist',
                      'road',
                      'parking',
                      'sidewalk',
                      'other-ground',
                      'building',
                      'fence',
                      'vegetation',
                      'trunk',
                      'terrain',
                      'pole',
                      'traffic-sign']

    fig = plt.figure(figsize=(7, 6))
    ax2 = plt.axes(projection="3d")
    sc = ax2.scatter(ptc[:, 0], ptc[:, 1], ptc[:, 2], s=0.1, color=color)
    ax2.set_zlim(256, 768)
    ax2.set_xlim(256, 768)
    ax2.set_ylim(256, 768)
    elev = 30
    azimut = -70
    ax2.view_init(elev, azimut)
    ax2.set_axis_off()

    # add class names
    recs = []
    for i in range(0, len(colorset)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colorset[i]))
    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig.subplots_adjust(bottom=0.1)
    name = name[:-4]
    plt.show()


def read_ply_files(filepath, only_geom=True, att_name="color"):
    '''
    Reads a ply file and returns a numpy array
        Parameters:
            filepath (string): path to the point cloud
            only_geom (boolean): reads only geometry
            att_name (string): name of the extra attribute that should be read
        Return:
            numpy_pc (np.ndarray): numpy array of shape (N, c) that contains
                                   the geometry and the extra attributes red
                                   from the file
    '''

    plydata = PlyData.read(os.path.join(filepath))

    data = [
        plydata['vertex'].data["x"],
        plydata['vertex'].data["y"],
        plydata['vertex'].data["z"],
    ]

    if not only_geom and att_name=="color":
        data += [
            plydata['vertex'].data["red"],
            plydata['vertex'].data["green"],
            plydata['vertex'].data["blue"],
        ]
    elif not only_geom and att_name != "color":
        data += [
            plydata['vertex'].data[att_name]
        ]
    numpy_pc = np.vstack([
        data
    ]).transpose()

    return numpy_pc


def write_ply_file(
    pc,
    filepath,
    ascii_text=False,
    attributes=None,
    dtype=None,
    names=None
):
    '''
    Writes out a point cloud as a ply file

        Parameters:
            pc (np.ndarray): point cloud ot be written out with shape(Npts, 3)
            filepath (string): path to the ply file
            ascii_text (bool): decides whether to write the file in
                               binary or ascii
            attributes (list): list of columns containing the various numpy 
                               attributes
            dtype (list): list of numpy dtpyes one for each attribute
            names (list): list of strings that are the name of the extra
                          attributes
    '''

    # transforming the pc in the format liked by
    if attributes is None:
        extra_col = []
        columns = []
        types = []
    else:
        extra_col = attributes.transpose()
        columns = [arr for arr in attributes.transpose()]
        types = [(names[i], dtype[i]) for i in range(len(names))]
    pc = np.array(
        list(zip(
            pc[:, 0].transpose(),
            pc[:, 1].transpose(),
            pc[:, 2].transpose(),
            *extra_col
        )),
        dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            *types
        ]
    )

    el = PlyElement.describe(pc, "vertex")

    with open(
        filepath, "wb"
    ) as pc_file:
        if ascii_text:
            PlyData([el], text=True).write(pc_file)
        else:
            PlyData([el], byte_order='<').write(pc_file)

def collapse_duplicate_points(pc):
    '''
    Takes a point cloud and collapses all overlapping points into 
    one by taking the average of the attributes
    Parameters:
        pc (np.ndarray): point cloud to be processed with shape (Npts, c) where
                         c is the number of channels (the first three channels
                         must be x, y, z)
    Return:
        out_pc: point cloud without duplicates
    '''
    
    input_dict = {"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2]}
    input_dict.update({i: x for i, x in enumerate(list(pc[:, 3:].transpose()))})
    pc_df = pd.DataFrame(input_dict)
    out_pc = pc_df.groupby(["x", "y", "z"]).mean().round().reset_index()
    return out_pc.to_numpy()

class Octree:

    def __init__(self, block: np.ndarray) -> None:
        resolution = block.shape[0]
        reshaped_pc = flatten_cubes(
            block,
            resolution,
            resolution//2
        )
        geom_presence = np.any(
            reshaped_pc[:, :, :, :, :1].reshape((2, 2, 2, -1)),
            axis = 3
        )
        self.children = [None] * 8
        self.color = None
        if resolution > 2:
            coords = list(np.array(np.where(geom_presence)).transpose())
            for x, y, z in coords:
                self.children[x + 2 * y + 4 * z] = Octree(
                    unflatten_cubes(
                        reshaped_pc[x, y, z],
                        resolution//2,
                        resolution//2
                    )
                )
        else:
            self.color = reshaped_pc.reshape((-1, 4))[:, 1:]
            
