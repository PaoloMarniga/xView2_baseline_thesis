#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          #
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, #
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################

import json
from shapely import wkt
import numpy as np
from cv2 import fillPoly, imwrite


def open_json(json_file_path):
    """
    :param json_file_path: path to open inference json file
    :returns: the json data dictionary of localized polygon and their classifications
    """
    with open(json_file_path) as jf:
        json_data = json.load(jf)
        inference_data = json_data['features']['xy']
        return inference_data


def create_image(inference_data):
    """
    :params inference_data: json data dictionary of localized polygon and their classifications
    :returns: a numpy RGB image with polygons filled in according to the key below

    Color key:
    0 black   = background
    1 green   = no-damage / un-classified
    2 yellow  = minor-damage
    3 orange  = major-damage
    4 red     = destroyed
    """
    color_key = {
        'un-classified': (0, 255, 0),
        'no-damage': (0, 255, 0),
        'minor-damage': (255, 255, 0),
        'major-damage': (255, 165, 0),
        'destroyed': (255, 0, 0),
    }

    # OpenCV uses BGR, not RGB
    bgr_color_key = {
        'un-classified': (0, 255, 0),
        'no-damage': (0, 255, 0),
        'minor-damage': (0, 255, 255),
        'major-damage': (0, 165, 255),
        'destroyed': (0, 0, 255),
    }

    mask_img = np.zeros((1024, 1024, 3), np.uint8)

    for poly in inference_data:
        damage = poly['properties'].get('subtype', 'un-classified')
        coords = wkt.loads(poly['wkt'])
        poly_np = np.array(coords.exterior.coords, np.int32)

        fillPoly(mask_img, [poly_np], bgr_color_key[damage])

    return mask_img


def save_image(polygons, output_path):
    """
    :param polygons: np array with filled in polygons from create_image()
    :param output_path: path to save the final output inference image
    """
    imwrite(output_path, polygons)


def create_inference_image(json_input_path, image_output_path):
    """
    :param json_input_path: Path to output inference json file
    :param image_output_path: Path to save the final inference image
    """
    inference_data = open_json(json_input_path)
    polygon_array = create_image(inference_data)
    save_image(polygon_array, image_output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="""inference_image_output.py: Takes the inference localization and classification final outputs in json and outputs a colored image."""
    )
    parser.add_argument(
        '--input',
        required=True,
        metavar='/path/to/final/inference.json',
        help="Full path to the final inference json"
    )
    parser.add_argument(
        '--output',
        required=True,
        metavar='/path/to/inference.png',
        help="Full path to save the image to"
    )

    args = parser.parse_args()

    create_inference_image(args.input, args.output)