import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import math
import os
import subprocess
from sys import platform
from plyfile import PlyData, PlyElement
from matplotlib.font_manager import FontProperties

def flatten_cubes(vol, nvx, nb):
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
    nbl = nvx // nb
    vol = vol.reshape(nbl, nbl, nbl, nb, nb, nb, -1)
    vol = vol.swapaxes(1, 2).swapaxes(3, 4).swapaxes(1, 4).reshape(
                nvx, nvx, nvx, -1
    )
    return vol




def qualityPCL(rec_ref, str_ref, scale=1023):
    if platform == "win32":
        command = os.path.join(
            "..",
            "evaluate",
            "test",
            "pc_error_0.09_win64.exe"
        ) + f"-a {str_ref}"
    else:
        command = os.path.join(
            "..",
            "evaluate",
            "test",
            "pc_error_d"
        ) + f" -a {str_ref}"
    command += f" -b {rec_ref} -d 1 -r {scale}"
    out = subprocess.check_output(command, shell=True).decode("utf-8")
    i = find_nth_occurrence(out, 'mseF', 2)
    psnrD1 = float(out[(i + 20):].split('\n')[0].strip())
    return psnrD1


def find_nth_occurrence(s, match, n):
    final_ind = 0
    for _ in range(n):
        i = s.find(match)
        s = s[(i + len(match)):]
        final_ind += i + len(match)
    return final_ind - len(match)


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
    elif only_geom:
        data += [
            np.zeros_like(plydata['vertex'].data["x"]),
            np.zeros_like(plydata['vertex'].data["x"]),
            np.zeros_like(plydata['vertex'].data["x"])
        ]
    else:
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


def encode_with_TMC13(
    path_to_TMC13,
    unc_data_path,
    comp_data_path,
    pc_scale_factor=1,
    silence_output=True,
    trisoup=False,
    q_level=0,
    ascii_text=False,
    encode_colors=False
):
    '''
    Uses TMC13 to encode a point cloud

        Parameters:
            path_to_TMC13 (string): path to the TMC13 executable
            unc_data_path (string): path to the PC to be compressed
            comp_data_path (string): path to the compressed file
            pc_scale_factor (float): how much to rescale the pc to obtain
                    almost lossless coding
            silence_output (bool): used to silence the output of the compression
                    command
            trisoup (bool):used to decide whether trisoup or standard
                    quantization should be used
            q_level (int): tells how many octree levels should be skipped both
                    with quantization and trisoup

    '''

    # building up the command
    command = " ".join([
        path_to_TMC13,
        f"--mode=0",
        f"--compressedStreamPath={comp_data_path}",
        f"--uncompressedDataPath={unc_data_path}",
        f"--inputScale={pc_scale_factor}",
        f"--externalScale={1/pc_scale_factor}"
    ])

    if encode_colors:
        command += " --attribute=color"
    if trisoup:
        command += f" --trisoupNodeSizeLog2={q_level + 1}"
    else:
        command += f" --codingScale={1/2**q_level}"

    if ascii_text:
        command += " --outputBinaryPly=0"

    if silence_output:
        command += " >/dev/null 2>&1"

    # executing the command
    os.system(command)


def decode_with_TMC13(
        compressed_path,
        reconstructed_path,
        path_to_TMC13,
        ascii_text=True,
        silence_output=True
):

    # building up the command
    command = " ".join([
        path_to_TMC13,
        f"--reconstructedDataPath={reconstructed_path}",
        f"--compressedStreamPath={compressed_path}",
        "--mode=1"

    ])

    if ascii_text:
        command += " --outputBinaryPly=0"

    if silence_output:
        command += " >/dev/null 2>&1"

    # executing the command
    os.system(command)


def decode_with_draco(
    compressed_path,
    reconstruction_path,
    path_to_draco,
    silence_output=True
):

    command = " ".join([
        os.path.join(path_to_draco, "draco_decoder"),
        f"-i {compressed_path}",
        f"-o {reconstruction_path}"
    ])

    if silence_output:
        command += " >/dev/null 2>&1"

    os.system(command)


def decode(
    compressed_path,
    reconstructed_path,
    path_to_codec,
    codec,
    ascii_text=True,
    silence_output=True
):
    if codec == "tmc13":
        decode_with_TMC13(
            compressed_path,
            reconstructed_path,
            path_to_codec,
            ascii_text=ascii_text,
            silence_output=silence_output
        )
    elif codec == "draco":
        decode_with_draco(
            compressed_path,
            reconstructed_path,
            path_to_codec,
            silence_output=silence_output
        )
    else:
        raise Exception(f"NotImplementedError: {codec} codec not supported")


def encode_with_draco(
    path_to_draco,
    input_path,
    compressed_path,
    quantization_bits=0,
    silence_output=True
):
    '''
    Uses draco to encode a point cloud
        Parameters:
            path_to_draco (string): path to the draco executable
            input_path (string): path to the point cloud to be coded
            compressed_path (string): path of the compressed representation
            quantization_bits (int): number of bits used to represent geometry
            silence_output (bool): used to silence the output of the compression
                                   command
    '''

    command = " ".join([
        os.path.join(path_to_draco, "draco_encoder"),
        f"-i {input_path}",
        f"-o {compressed_path}",
        f"-qp {15 - quantization_bits}",
        f"-point_cloud"
    ])

    if silence_output:
        command += " >/dev/null 2>&1"

    os.system(command)


def encode(
    path_to_codec,
    input_path,
    compressed_path,
    quantization_bits=10,
    silence_output=True,
    codec="draco",
    **args
):
    '''
    Compress a point cloud
        Parameters:
            path_to_codec (string): path to the codec executable
            input_path (string): path to the point cloud to be coded
            compressed_path (string): path of the compressed representation
            output (string): path to the reconstructed point cloud
            quantization_bits (int): number of bits used to represent geometry
            silence_output (bool): used to silence the output of the compression
                    command
            codec (string): used to choose which codec to use (TMC13/draco)
            **args: dictionary of parameters for tmc13 (see function
                    encode_with_TMC13)
    '''

    if codec == "draco":
        encode_with_draco(
            path_to_codec,
            input_path,
            compressed_path,
            quantization_bits=quantization_bits,
            silence_output=silence_output
        )

    elif codec == "tmc13":
        encode_with_TMC13(
            path_to_codec,
            input_path,
            compressed_path,
            q_level=quantization_bits,
            silence_output=silence_output,
            **args
        )
    else:
        raise Exception(f"NotImplementedError: {codec} codec not supported")


def encode_and_decode(
    path_to_codec,
    input_path,
    compressed_path,
    reconstructed_path,
    quantization_bits=10,
    silence_output=True,
    codec="draco",
    **args
):
    encode(
        path_to_codec,
        input_path,
        compressed_path,
        quantization_bits=quantization_bits,
        silence_output=silence_output,
        codec=codec,
        **args
    )

    decode(
        compressed_path,
        reconstructed_path,
        path_to_codec,
        codec,
        ascii_text=silence_output,
    )

# taken from
# https://github.com/jascenso/bjontegaard_metrics/blob/master/bj_delta.py
def bj_delta(R1, PSNR1, R2, PSNR2, mode=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # find integral
    if mode == 0:
        # least squares polynomial fit
        p1 = np.polyfit(lR1, PSNR1, 3)
        p2 = np.polyfit(lR2, PSNR2, 3)

        # integration interval
        min_int = max(min(lR1), min(lR2))
        max_int = min(max(lR1), max(lR2))

        # indefinite integral of both polynomial curves
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        # evaluates both poly curves at the limits of the integration interval
        # to find the area
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

        # find avg diff between the areas to obtain the final measure
        avg_diff = (int2-int1)/(max_int-min_int)
    else:
        # rate method: sames as previous one but with inverse order
        p1 = np.polyfit(PSNR1, lR1, 3)
        p2 = np.polyfit(PSNR2, lR2, 3)

        # integration interval
        min_int = max(min(PSNR1), min(PSNR2))
        max_int = min(max(PSNR1), max(PSNR2))

        # indefinite interval of both polynomial curves
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        # evaluates both poly curves at the limits of the integration interval
        # to find the area
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

        # find avg diff between the areas to obtain the final measure
        avg_exp_diff = (int2-int1)/(max_int-min_int)
        avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff
