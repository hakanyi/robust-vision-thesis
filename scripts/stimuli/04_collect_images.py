""" This is a helper script to do the following:
1. Take a base and change mesh from a pool of meshes
2. Sample lamp_angles to make base, mesh-change, and lamp-change mooneys.
3. For each sample, record lamp angles pixel distances and pixel distances."""
import os
import os.path as osp
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import glob
import json

import robust_vision.utils.workspace as ws
import robust_vision.utils.metrics as metrics
import trimesh


def get_arguments():
    arg_parser = argparse.ArgumentParser(
        description="Make change detection stimuli (images) with SMPL bodies"
    )
    arg_parser.add_argument(
        "--exp-dir", "-e", dest="exp_dir", required=True,
        help="Experiment directory.",
    )
    args = arg_parser.parse_args()
    return args

def read_postures_file(pth):
    with open(pth) as f:
        pairs = f.read().splitlines()
    postures = [str.split(pair, ' ') for pair in pairs]
    postures = pd.DataFrame(postures, columns=["base", "change"])
    return postures

def central_angle(angles1, angles2, deg=True):
    angles1, angles2 = np.array(angles1), np.array(angles2)
    if deg:
        angles1, angles2 = angles1 / 180 * np.pi, angles2 / 180 * np.pi
    long1, lat1 = angles1
    long2, lat2 = angles2
    d_long, d_lat = long1 - long2, lat1 - lat2

    # haversine formula:
    # https://en.wikipedia.org/wiki/Great-circle_distance#Formulae
    under_root = np.sin(d_lat/2)**2 + \
        np.cos(lat1)*np.cos(lat2)*np.sin(d_long/2)**2
    centr_angle = 2 * np.arcsin(np.sqrt(under_root))
    return centr_angle

def main():
    args = get_arguments()

    # directory where we'll save the data
    stim_dir = osp.join(args.exp_dir, ws.STIMULI_DIR)
    img_dir = osp.join(stim_dir, "images")
    df_pth = osp.join(img_dir, "images_output.csv")
    os.makedirs(img_dir, exist_ok=True)

    # load experiment specifications
    specs = ws.load_experiment_specifications(args.exp_dir)

    # load h36m data for pose and shape ground truths
    h36m_data = np.load(specs["h36mPath"])
    imgnames = [osp.split(name)[-1] for name in h36m_data['imgname']]

    csv_folders = specs["csv_folders"]

    out_df = pd.DataFrame()
    counter = 0
    for csv_folder in tqdm(csv_folders):
        # read posture file
        sample_dir = osp.join(args.exp_dir, ws.STIMULI_DIR, csv_folder)
        postures_df = read_postures_file(osp.join(sample_dir,
                                                  "base_postures.txt"))
        # read the csv file
        final_df = pd.read_csv(osp.join(sample_dir, "final_output.csv"))
        final_df["lamp_angle_base"] = final_df["lamp_angle_base"].apply(json.loads)
        final_df["lamp_angle_changed"] = final_df["lamp_angle_changed"].apply(json.loads)
        for idx, row in tqdm(final_df.iterrows()):
            # copy the corresponding images to the target folder
            old_name = f"{row.scene:02d}_{row.sample_id}"
            for f in glob.glob(osp.join(sample_dir, "images", f"{old_name}*")):
                suffix = str.split(osp.split(f)[1], "_")[2:]
                new_name = "_".join([f"{counter:02d}", *suffix])
                shutil.copy(f, osp.join(img_dir, new_name))

            # record ground truth data:
            # 1. light distance in deg
            light_distance = central_angle(row["lamp_angle_base"],
                                           row["lamp_angle_changed"])
            light_distance = light_distance / np.pi * 180

            # 2. pose and shape distance
            postures = postures_df.iloc[row.scene]

            idx_base = imgnames.index(postures["base"]+".jpg")
            idx_changed = imgnames.index(postures["change"]+".jpg")

            pose_distance = (h36m_data['pose'][idx_base][3:] -
                            h36m_data['pose'][idx_changed][3:])**2
            shape_distance = (h36m_data['shape'][idx_base][3:] -
                            h36m_data['shape'][idx_changed][3:])**2

            # 3. mesh distance
            pth = osp.join(sample_dir, "meshes", postures["base"]+".obj")
            mesh_base = trimesh.load(pth, process=False)
            pth = osp.join(sample_dir, "meshes", postures["change"]+".obj")
            mesh_changed = trimesh.load(pth, process=False)
            mesh_distance = metrics.chamfer_distance(mesh_base, mesh_changed)

            # update the row
            new_row = row.drop("sample_id")
            new_row['scene'] = counter
            new_row['light-distance'] = light_distance
            new_row['mesh-distance'] = mesh_distance
            new_row['posevec-distance'] = pose_distance.mean()
            new_row['shapevec-distance'] = shape_distance.mean()
            new_row = new_row.rename({
                "pixel_distance_light" : "pixel_distance-different_light",
                "pixel_distance_posture" : "pixel_distance-different_pose",
                "scene" : "pair_id"
            })
            out_df = out_df.append(new_row, ignore_index=True)

            counter += 1

    out_df.to_csv(df_pth, index=False)
    print(f"Saved info as {df_pth}.")

if __name__ == "__main__":
    main()
