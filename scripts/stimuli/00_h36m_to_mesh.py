import os
import os.path as osp
import argparse
import numpy as np
import torch
import trimesh
import pandas as pd
from tqdm import tqdm

import robust_vision.utils.workspace as ws
from robust_vision.models import SMPL

"""
Take a txt file where each line has two things:
1. The name of the h36m image with a body in BASE posture
2. The name of the h36m image with a body with CHANGE posture
Produce meshes that corresponds to all these images and save them.
"""

ZERO_GLOBAL_ORIENT = True

def get_arguments():
    arg_parser = argparse.ArgumentParser(
        description="Make meshes (.obj) from the corrresponding h36m images."
    )
    arg_parser.add_argument(
        "--exp-dir", "-e", dest="exp_dir", required=True,
        help="Experiment directory.",
    )
    arg_parser.add_argument(
        "--sample-dir", "-sd", dest="sample_dir", required=True,
        help="Directory within the experiment directory that " + \
        " contains a base_postures.txt",
    )
    args = arg_parser.parse_args()
    return args

def read_postures_file(pth):
    with open(pth) as f:
        pairs = f.read().splitlines()
    postures = [str.split(pair, ' ') for pair in pairs]
    postures = pd.DataFrame(postures, columns=["base", "change"])
    return postures

if __name__ == "__main__":
    args = get_arguments()

    # our sample directory
    sample_dir = osp.join(args.exp_dir, ws.STIMULI_DIR, args.sample_dir)

    # load experiment specifications
    specs = ws.load_experiment_specifications(args.exp_dir)

    # load h36m dataset
    h36m_data = np.load(specs["h36mPath"])
    imgnames = [osp.split(name)[-1] for name in h36m_data['imgname']]

    # read postures file
    pth = osp.join(sample_dir, 'base_postures.txt')
    postures_df = read_postures_file(pth)

    # directory where we'll save the meshes
    mesh_dir = osp.join(sample_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # load SMPL
    smpl = SMPL(specs["smplDir"], batch_size=1, create_transl=False)
    smpl = smpl.cuda()

    for idx, row in tqdm(postures_df.iterrows()):
        for key in ["base", "change"]:
            h36m_idx = imgnames.index(row[key]+".jpg")
            pose = torch.from_numpy(h36m_data["pose"][h36m_idx]).float().unsqueeze(0).cuda()
            shape = torch.from_numpy(h36m_data["shape"][h36m_idx]).float().unsqueeze(0).cuda()

            if ZERO_GLOBAL_ORIENT:
                go = torch.zeros((1, 3), device="cuda")
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(90), [0, 1, 0])
            else:
                go = pose[:, :3]
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(180), [1, 0, 0])

            out = smpl(body_pose=pose[:, 3:], global_orient=go, betas=shape)
            mesh = trimesh.Trimesh(out.vertices.cpu().squeeze(0), smpl.faces)
            mesh.apply_transform(rot)
            pth = osp.join(mesh_dir, row[key]+".obj")
            mesh.export(pth)
