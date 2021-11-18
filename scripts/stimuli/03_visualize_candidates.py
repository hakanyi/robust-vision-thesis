""" This is a helper script to do the following:
1. Take a base and change mesh from a pool of meshes
2. Sample lamp_angles to make base, mesh-change, and lamp-change mooneys.
3. For each sample, record lamp angles pixel distances and pixel distances."""
import json
import os
import os.path as osp
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from robust_vision.renderer import Renderer
import robust_vision.utils.workspace as ws
import robust_vision.utils.mitsuba as umitsuba
from torchvision.transforms.functional import to_pil_image
from PIL import Image

def get_arguments():
    arg_parser = argparse.ArgumentParser(
        description="Visualize stimulus candidates from a candidate_pairs.csv."
    )
    arg_parser.add_argument(
        "--exp-dir", "-e", dest="exp_dir", required=True,
        help="Experiment directory.",
    )
    arg_parser.add_argument(
        "--sample-dir", "-sd", dest="sample_dir", required=True,
        help="Name of the directory that holds the candidate_pairs.csv file.",
    )
    arg_parser.add_argument(
        "--max-per-scene", "-mps", dest="max_per_scene",
        default=5, type=int,
        help="Maximum number of pairs to render."
    )
    args = arg_parser.parse_args()
    return args

def create_img(renderer, lamp_angle, spp, save_name=None):
    renderer.rotate_lamp(lamp_angle[0], lamp_angle[1])
    renderer.render_scene(spp)
    shaded = to_pil_image(renderer.read_output("shading").cpu())
    mooney = to_pil_image(renderer.read_output("mooney").cpu())
    if save_name is not None:
        shaded.save(f"{save_name}_shaded.png")
        mooney.save(f"{save_name}_mooney.png")
    return mooney, shaded

def image_distance(img1, img2):
    img1 = np.array(img1).astype(float)
    img2 = np.array(img2).astype(float)
    dist = np.abs(img1-img2).mean()
    return dist

def make_diff_image(base, diff, pth):
    base = np.array(base).astype(float)
    diff = np.array(diff).astype(float)
    res = np.repeat(diff[:, :, np.newaxis], 3, axis=2)
    res[diff-base > 0, :] = [0,255,0]
    res[diff-base < 0, :] = [255,0,0]
    Image.fromarray(np.uint8(res)).save(pth+".png")

def initialize_renderer(specs):
    placeholder_pth = osp.join(specs['MeshDir'], "placeholder.obj")
    renderer = Renderer(
        lamp_radiance=specs["LampRadiance"], lamp_origin=specs["LampOrigin"],
        lamp_up=specs["LampUpVector"], max_depth=specs["RendererMaxDepth"],
        file_pth=placeholder_pth, res=specs["ImageResolution"],
        cam_azimuth=specs["CameraAzimuth"],
        cam_elevation=specs["CameraElevation"],
        cam_distance=specs["CameraDistance"],
        cam_translation=specs["CameraTranslation"],
    )
    return renderer

def read_postures_file(pth):
    with open(pth) as f:
        pairs = f.read().splitlines()
    postures = [str.split(pair, ' ') for pair in pairs]
    postures = pd.DataFrame(postures, columns=["base", "change"])
    return postures

def main():
    args = get_arguments()

    # directory where we'll save the data
    sample_dir = osp.join(args.exp_dir, ws.STIMULI_DIR, args.sample_dir)
    df_pth = osp.join(sample_dir, "candidate_output.csv")

    img_dir = osp.join(sample_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # load experiment specifications
    specs = ws.load_experiment_specifications(args.exp_dir)

    # read postures file
    pth = osp.join(sample_dir, 'base_postures.txt')
    postures_df = read_postures_file(pth)
    # postures_df = postures_df.iloc[args.scene_id]
    spp = specs["RendererSamplesPerPixel"]

    # read candidate_pairs.csv file
    candidate_df = pd.read_csv(osp.join(sample_dir, "candidate_pairs.csv"))
    candidate_df["lamp_angle_base"] = candidate_df["lamp_angle_base"].apply(json.loads)
    candidate_df["lamp_angle_changed"] = candidate_df["lamp_angle_changed"].apply(json.loads)
    scenes = candidate_df.scene.unique()

    # initialize renderer
    renderer = initialize_renderer(specs)

    data = []

    def make_pth(sample_id, suffix):
        return osp.join(img_dir, f"{scene:02d}_{sample_id}_{suffix}")

    for scene in tqdm(scenes):
        scene_df = candidate_df[candidate_df.scene == scene].reset_index()
        meshes = postures_df.iloc[scene]
        for idx, row in scene_df.iterrows():
            if idx == args.max_per_scene:
                break

            lamp_angle_base = row["lamp_angle_base"]
            lamp_angle_changed = row["lamp_angle_changed"]

            # load scene with base obj
            pth = osp.join(sample_dir, "meshes", meshes["base"]+".obj")
            verts_base, faces_base = umitsuba.load_obj(pth)
            renderer.replace_mesh(verts_base, faces_base)
            save_pth = make_pth(row['sample_id'], "base")
            mooney1, _ = create_img(renderer, lamp_angle_base, spp, save_pth)
            save_pth = make_pth(row['sample_id'], "light")
            mooney2, _ = create_img(renderer, lamp_angle_changed, spp, save_pth)
            pixel_distance_light = image_distance(mooney1, mooney2)
            save_pth = make_pth(row['sample_id'], "light-diff")
            make_diff_image(mooney1, mooney2, save_pth)

            # load scene with change obj
            pth = osp.join(sample_dir, "meshes", meshes["change"]+".obj")
            verts_changed, faces_changed = umitsuba.load_obj(pth)
            renderer.replace_mesh(verts_changed, faces_changed)
            save_pth = make_pth(row['sample_id'], "posture")
            mooney2, _ = create_img(renderer, lamp_angle_base, spp, save_pth)
            pixel_distance_pose = image_distance(mooney1, mooney2)
            save_pth = make_pth(row['sample_id'], "posture-diff")
            make_diff_image(mooney1, mooney2, save_pth)

            dist_ratio = pixel_distance_light/pixel_distance_pose
            data.append({
                "scene": scene,
                "sample_id": row['sample_id'],
                "lamp_angle_base": lamp_angle_base,
                "lamp_angle_changed": lamp_angle_changed,
                "pixel_distance_light": pixel_distance_light,
                "pixel_distance_posture": pixel_distance_pose,
                "pixel_distance_ratio": dist_ratio,
            })

    pd.DataFrame.from_records(data).to_csv(df_pth, index=False)
    print(f"Saved info as {df_pth}.")

if __name__ == "__main__":
    main()
