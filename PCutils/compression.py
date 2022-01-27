import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import math
import os
import subprocess
from sys import platform
from plyfile import PlyData, PlyElement
from matplotlib.font_manager import FontProperties
from sklearn.neighbors import KDTree
import pandas as pd

def qualityPCL(rec_ref, str_ref, scale=1023, colored=False):
    '''
    Used to compute the D1 and D2 metrics for two PCs
    Parameters:
        rec_ref (string): path to the reconstructed PC
        str_ref (string): path to the original PC
        scale (float): reference scale
        colored (boolean): wether to compute also Yuv PSNR
    Returns:
        psnrD1 (float): D1 psnr
    '''
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

    if colored:
        command += " -c 1"

    out = subprocess.check_output(command, shell=True).decode("utf-8")

    if not colored:
        i = _find_nth_occurrence(out, 'mseF', 2)
        psnrD1 = float(out[(i + 20):].split('\n')[0].strip())
        return psnrD1

    else:
        i = _find_nth_occurrence(out, 'mseF', 2)
        psnrD1 = float(out[(i + 20):(i + 27)].strip())
        i = out.find('h.,PSNR')
        psnrD1H = float(out[(i + 20):(i + 27)].strip())
        i = _find_nth_occurrence(out, 'PSNRF', 1)
        psnrY = float(out[(i + 15):(i + 22)].strip())
        i = _find_nth_occurrence(out, 'PSNRF', 2)
        psnrU = float(out[(i + 15):(i + 22)].strip())
        i = _find_nth_occurrence(out, 'PSNRF', 3)
        temp = out[(i + 15):(i + 22)].strip()
        if "inf" in temp:
            psnrV = np.inf
        else:
            psnrV = float(temp)
        return [psnrD1, psnrD1H, psnrY, psnrU, psnrV]

def _find_nth_occurrence(s, match, n):
    """
    gets the nth occurrence of match in s
    Parameters:
        s (string): full string
        match (string): patter to match
        n (int): number of times the pattern should be matched
    Return:
        index (int): index in the string of the first charachter of the nth
                     match
    """
    final_ind = 0
    for _ in range(n):
        i = s.find(match)
        s = s[(i + len(match)):]
        final_ind += i + len(match)
    return final_ind - len(match)

def encode_with_TMC13(
    path_to_TMC13,
    unc_data_path,
    comp_data_path,
    pc_scale_factor=1,
    silence_output=True,
    trisoup=False,
    q_level=0,
    ascii_text=False,
    encode_colors=False,
    **args
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
            ascii_text (Bool): wether the reconstructed PC should be in binary
                               or ascii form
            encode_colors (Bool): if true also colors are encoded
            **args: extra arguments specifics to TMC13 (eg. qp that sets 
                    quantization parameter for the Y component)
        Return:
            execution time in milliseconds
    '''

    tmc13_args = {
        "mode": 0,
        "compressedStreamPath": comp_data_path,
        "uncompressedDataPath": unc_data_path,
        "inputScale": pc_scale_factor,
        "externalScale": 1/pc_scale_factor,
    }

    if encode_colors:
        tmc13_args["attribute"] = "color"


    if trisoup:
        tmc13_args["trisoupNodeSizeLog2"] = q_level + 1
    else:
        tmc13_args["codingScale"] = 1/2**q_level

    if ascii_text:
        tmc13_args["outputBinaryPly"] = 0

    tmc13_args.update(args)
    return _encode_with_TMC13(path_to_TMC13, silence_output, **tmc13_args)


def _encode_with_TMC13(TMC13, silence_output=True, **args):

    '''
    Uses TMC13 to encode a point cloud with arbitrary parameters

        Parameters:
            TMC13 (string): path to the TMC13 executable
            silence_output (bool): used to silence the output of the compression
                    command
            **args: arguments specifics to TMC13 (eg. qp that sets 
                    quantization parameter for the Y component)
        Return:
            execution time in milliseconds
    '''

    command = " ".join(
        [TMC13] + 
        [f"--{key}={args[key]}" for key in args]
    )

    command += " 2>&1"

    # executing the command
    f = os.popen(command, "r")
    output = f.read()
    match_string = "Processing time (user):"
    start_index = output.find(match_string) + len(match_string)
    end_index = output.find("s", start_index)
    time = float(output[start_index:end_index-1])
    if not silence_output:
        print(output)

    return time

def decode_with_TMC13(
        compressed_path,
        reconstructed_path,
        path_to_TMC13,
        ascii_text=True,
        silence_output=True
):


    '''
    Uses TMC13 to decode a point cloud

        Parameters:
            compressed_path (string): path to the compressed PC
            reconstructed_path (string): path where the PC should be 
                                         reconstructed
            path_to_TMC13 (string): path to the TMC13 executable
            ascii_text (Bool): wether the reconstructed PC should be in binary
                               or ascii form
            silence_output (bool): used to silence the output of the compression
                    command
        Return:
            execution time in milliseconds

    '''
    # building up the command
    command = " ".join([
        path_to_TMC13,
        f"--reconstructedDataPath={reconstructed_path}",
        f"--compressedStreamPath={compressed_path}",
        "--mode=1"

    ])

    if ascii_text:
        command += " --outputBinaryPly=0"

    command += " 2>&1"

    # executing the command
    f = os.popen(command, "r")
    output = f.read()
    match_string = "Processing time (user):"
    start_index = output.find(match_string) + len(match_string)
    end_index = output.find("s", start_index)
    if not silence_output:
        print(output)

    time = float(output[start_index:end_index-1])
    return time



def decode_with_draco(
    compressed_path,
    reconstruction_path,
    path_to_draco,
    silence_output=True
):

    '''
    Uses Draco to decode a point cloud

        Parameters:
            compressed_path (string): path to the compressed PC
            reconstructed_path (string): path where the PC should be 
                                         reconstructed
            path_to_draco (string): path to the draco executable
            silence_output (bool): used to silence the output of the compression
                    command
        Return:
            execution time in milliseconds

    '''

    command = " ".join([
        os.path.join(path_to_draco, "draco_decoder"),
        f"-i {compressed_path}",
        f"-o {reconstruction_path}"
    ])

    command += " 2>&1"

    # executing the command
    f = os.popen(command, "r")
    output = f.read()
    match_string = "ms to decode"
    end_index = output.find(match_string) 
    start_index = output.find("(", end_index - 10) + 1
    time = float(output[start_index:end_index]) / 1000
    if not silence_output:
        print(output)

    return time


def decode(
    compressed_path,
    reconstructed_path,
    path_to_codec,
    codec,
    ascii_text=True,
    silence_output=True
):
    '''
    Uses either TMC13 or draco to decode a point cloud

        Parameters:
            compressed_path (string): path to the compressed PC
            reconstructed_path (string): path where the PC should be 
                                         reconstructed
            path_to_codec (string): path to the codec executable
            codec (string): name of the codec (either draco or tmc13)
            ascii_text (Bool): wether the reconstructed PC should be in binary
                               or ascii form
            silence_output (bool): used to silence the output of the compression
                    command
        Return:
            execution time in milliseconds

    '''
    if codec == "tmc13":
        return decode_with_TMC13(
            compressed_path,
            reconstructed_path,
            path_to_codec,
            ascii_text=ascii_text,
            silence_output=silence_output
        )
    elif codec == "draco":
        return decode_with_draco(
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
        Return:
            execution time in milliseconds
    '''

    command = " ".join([
        os.path.join(path_to_draco, "draco_encoder"),
        f"-i {input_path}",
        f"-o {compressed_path}",
        f"-qp {15 - quantization_bits}",
        f"-point_cloud"
    ])

    command += " 2>&1"

    # executing the command
    f = os.popen(command, "r")
    output = f.read()
    match_string = "ms to encode"
    end_index = output.find(match_string) 
    start_index = output.find("(", end_index - 10) + 1
    time = float(output[start_index:end_index]) / 1000
    if not silence_output:
        print(output)

    return time


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
            quantization_bits (int): number of bits used to represent geometry
            silence_output (bool): used to silence the output of the compression
                    command
            codec (string): used to choose which codec to use (tmc13/draco)
            **args: dictionary of parameters for tmc13 (see function
                    encode_with_TMC13)
        Return:
            execution time in milliseconds
    '''

    if codec == "draco":
        return encode_with_draco(
            path_to_codec,
            input_path,
            compressed_path,
            quantization_bits=quantization_bits,
            silence_output=silence_output
        )

    elif codec == "tmc13":
        return encode_with_TMC13(
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
    '''
    encodes and decodes a point cloud
        Parameters:
            path_to_codec (string): path to the codec executable
            input_path (string): path to the point cloud to be coded
            compressed_path (string): path of the compressed representation
            reconstructed_path (string): path where the PC will be 
                                         reconstructed
            quantization_bits (int): number of bits to be removed via 
                                     quantization
            silence_output (bool): used to silence the output of the compression
                    command
            codec (string): used to choose which codec to use (tmc13/draco)
            **args: dictionary of parameters for tmc13 (see function
                    encode_with_TMC13)
        Return:
            execution time in milliseconds
    '''
    time_encode = encode(
        path_to_codec,
        input_path,
        compressed_path,
        quantization_bits=quantization_bits,
        silence_output=silence_output,
        codec=codec,
        **args
    )

    time_decode = decode(
        compressed_path,
        reconstructed_path,
        path_to_codec,
        codec,
        ascii_text=silence_output,
    )

    return time_encode + time_decode

# taken from
# https://github.com/jascenso/bjontegaard_metrics/blob/master/bj_delta.py
def bj_delta(R1, PSNR1, R2, PSNR2, mode=0, poly_exp=3):
    '''
    Computes the Bjontegaard delta rate for the given plots
    Parameters:
        R1 (np.ndarray): rates for the new codec
        PSNR1 (np.ndarray): PSNR for the new codec
        R1 (np.ndarray): rate for the reference codec
        PSNR1 (np.ndarray): PSNR for the reference codec
        mode (int): 0 returns delta rate 1 returns delta PSNR
        poly_exp (int): defines the degree of the fitted polynomial 
    Returns:
        avg_diff (float): delta value
        

    '''
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # find integral
    if mode == 0:
        # least squares polynomial fit
        p1 = np.polyfit(lR1, PSNR1, poly_exp)
        p2 = np.polyfit(lR2, PSNR2, poly_exp)

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
        p1 = np.polyfit(PSNR1, lR1, poly_exp)
        p2 = np.polyfit(PSNR2, lR2, poly_exp)

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

def one_level_raht(block: np.ndarray) -> np.ndarray:

    '''
    performs a slightly different raht transform
    Parameters:
        block: array of 2x2x2 blocks to be transformed
    Returns:
        the transformed 2x2x2 block
    '''

    geom = block[..., :1]
    lf = block[..., 1:] * geom

    w = geom
    final = []
    for i in range(3):

        w_sqrt = np.sqrt(w) 
        w_sqrt_inv = np.concatenate([
            - w_sqrt[(slice(None),) * (i + 1) + (slice(1, 2),)],
            w_sqrt[(slice(None),) * (i + 1) + (slice(0, 1),)],
        ], axis = i + 1)
        w_coeffs = np.concatenate([w_sqrt, w_sqrt_inv], axis=-1)
        w_coeffs = np.swapaxes(w_coeffs, i+1, -2).swapaxes(-1, -2)
        
        w_div = np.maximum(np.sum(w_sqrt, axis = i+1, keepdims=True), 1)
        w = np.sum(w, axis = i+1, keepdims=True)

        coeffs = w_coeffs @ np.swapaxes(lf, i + 1, -2)
        coeffs = np.swapaxes(coeffs, i+1, -2) / w_div
        hf = coeffs[(slice(None),)*(i + 1) + (slice(1, 2),)]
        lf = coeffs[(slice(None),)*(i + 1) + (slice(0, 1),)]
        final = [hf.reshape((
            block.shape[0],
            -1,
            3
        ))] + final
    final = [lf.reshape((
        block.shape[0],
        -1,
        3
    ))] + final

    return np.concatenate(final, axis=1)

def one_level_inverse_raht(
    coeffs: np.ndarray,
    geometry: np.ndarray
) -> np.ndarray:

    '''
    performs the invers operation of one_level_raht
    Parameters:
        coeffs: array of raht coefficients
        geometry: geometry information used to compute the coefficients
    Returns:
        the original blocks
    '''

    lf = coeffs[:, :1].reshape((-1, 1, 1, 1, 3))
    for i in range(3):

        w = geometry
        for j in range(3 - i - 1):
            w = np.sum(w, axis = j + 1, keepdims=True)

        w_sqrt = np.sqrt(w) 
        w_sqrt_inv = np.concatenate([
            - w_sqrt[(slice(None),) * (3-i) + (slice(1, 2),)],
            w_sqrt[(slice(None),) * (3-i) + (slice(0, 1),)],
        ], axis = 3 - i)
        w_coeffs = np.concatenate([w_sqrt, w_sqrt_inv], axis=-1)
        w_coeffs = np.swapaxes(w_coeffs, 3-i, -2).swapaxes(-1, -2)
        zero_mat = np.abs(w_coeffs).reshape(w_coeffs.shape[:-2] + (4,))
        zero_mat = np.where(zero_mat.sum(axis=-1) == 0)
        w_coeffs[zero_mat] = np.eye(2).reshape((1, 1, 1, 2, 2))
        w_coeffs = np.linalg.inv(w_coeffs)
        w_coeffs[zero_mat] = 0
        
        w_div = np.sum(w_sqrt, axis = 3-i, keepdims=True)

        hf = coeffs[:, 2**i:2**(i+1)] 
        mixed_coeffs = np.concatenate([
            lf,
            hf.reshape(lf.shape) 
        ], axis=3-i).swapaxes(3-i, -2)
        lf = (w_coeffs @ mixed_coeffs).swapaxes(3-i, -2) * w_div

    return lf

def D1_PSNR(pc1, pc2, scale=None):

    '''
    Parameters:
        pc1 (np.ndarray): geometry of the first point cloud, 
                          must have shape (n, 3) where n is the number of points
        pc2 (np.ndarray): geometry of the second point cloud, 
                          must have shape (n, 3) where n is the number of points
        scale (int): scale factor to be used in the psnr calculation, if None the 
                     scale factor is chosen as the maximum coordinate excursion 
                     of pc1
    Returns (int): the D1 PSNR as calculated in qualityPCL but faster
    '''

    if scale is None:
        scale = np.max(np.max(pc1, axis = 0) - np.min(pc1, axis = 0))

    # finding out distance between points in pc2 and 
    # their nn in pc1
    tree1 = KDTree(pc1, leaf_size=32)
    dists1 = tree1.query(pc2, k=1)[0]
    # the / 3 is because probably in the mpeg script they just compute 
    # mean squared error instead of the average of the squared distance
    dist1 = (dists1 ** 2).mean() / 3
    
    # finding out distance between points in pc1 and 
    # their nn in pc2
    tree2 = KDTree(pc2, leaf_size=32)
    dists2 = tree2.query(pc1, k=1)[0]
    # the / 3 is because probably in the mpeg script they just compute 
    # mean squared error instead of the average of the squared distance
    dist2 = (dists2 ** 2).mean() / 3

    return 10 * np.log10(scale**2/max(dist1, dist2))
