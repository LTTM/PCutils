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
    _encode_with_TMC13(path_to_TMC13, silence_output, **tmc13_args)


def _encode_with_TMC13(TMC13, silence_output=True, **args):

    '''
    Uses TMC13 to encode a point cloud with arbitrary parameters

        Parameters:
            TMC13 (string): path to the TMC13 executable
            silence_output (bool): used to silence the output of the compression
                    command
            **args: arguments specifics to TMC13 (eg. qp that sets 
                    quantization parameter for the Y component)
    '''

    command = " ".join(
        [TMC13] + 
        [f"--{key}={args[key]}" for key in args]
    )

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

    '''
    Uses Draco to decode a point cloud

        Parameters:
            compressed_path (string): path to the compressed PC
            reconstructed_path (string): path where the PC should be 
                                         reconstructed
            path_to_draco (string): path to the draco executable
            silence_output (bool): used to silence the output of the compression
                    command

    '''

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

    '''
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
            quantization_bits (int): number of bits used to represent geometry
            silence_output (bool): used to silence the output of the compression
                    command
            codec (string): used to choose which codec to use (tmc13/draco)
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
    '''
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
    col = block[..., 1:] * geom

    w = geom
    final = []
    for i in range(3):

        w_sqrt = np.sqrt(w) 
        w_div = np.maximum(np.sum(w_sqrt, axis = 1), 1)
        w = np.sum(w, axis = 1)

        lf = np.sum(col * w_sqrt, axis = 1) / w_div
        hf = (- w_sqrt[:, 1:] * col[:, :1] + \
                w_sqrt[:, :1] * col[:, 1:]).squeeze(1) / w_div
        col = lf
        final = [hf.reshape((
            block.shape[0],
            -1,
            block.shape[-1] - 1
        ))] + final
    final = [lf.reshape((
        block.shape[0],
        -1,
        block.shape[-1] - 1
    ))] + final

    return np.concatenate(final, axis=1)