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

from robust_vision.renderer import Renderer
import robust_vision.utils.workspace as ws
import robust_vision.utils.mitsuba as umitsuba
from torchvision.transforms.functional import to_pil_image


LAMP_MOVE_AZIMUTH_MIN = -40
LAMP_MOVE_AZIMUTH_MAX = 40
LAMP_MOVE_ELEVATION_MIN = -10
LAMP_MOVE_ELEVATION_MAX = 40

def get_arguments():
    arg_parser = argparse.ArgumentParser(
        description="Make change detection stimuli (images) with SMPL bodies"
    )
    arg_parser.add_argument(
        "--exp-dir", "-e", dest="exp_dir", required=True,
        help="Experiment directory.",
    )
    arg_parser.add_argument(
        "--sample-dir", "-sd", dest="sample_dir", required=True,
        help="Name of the directory to save the data in.",
    )
    arg_parser.add_argument(
        "--scene_id", required=True, type=int, help="Scene id."
    )
    arg_parser.add_argument(
        "--num-samples", "-n", dest="num_samples", default=10,
        type=int, help="Number of base lamp angles to sample."
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

def sample_lamp_rotation(azimuth_min=-30, azimuth_max=30,
                         elevation_min=-30, elevation_max=30):
    azimuth = np.random.uniform(azimuth_min, azimuth_max)
    elevation = np.random.uniform(elevation_min, elevation_max)
    return [azimuth, elevation]

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
    df_pth = osp.join(sample_dir, f"{args.scene_id:02d}.csv")

    # load experiment specifications
    specs = ws.load_experiment_specifications(args.exp_dir)

    # read postures file
    pth = osp.join(sample_dir, 'base_postures.txt')
    postures_df = read_postures_file(pth)
    if args.scene_id >= len(postures_df):
        print("The scene ID given is larger than the number of scenes.")
        return()
    postures_df = postures_df.iloc[args.scene_id]
    spp = specs["RendererSamplesPerPixel"]

    # initialize renderer
    renderer = initialize_renderer(specs)

    data = []

    for n in tqdm(range(args.num_samples), desc="rendering and recording"):
        lamp_angle_base = sample_lamp_rotation(LAMP_MOVE_AZIMUTH_MIN,
                                               LAMP_MOVE_AZIMUTH_MAX,
                                               LAMP_MOVE_ELEVATION_MIN,
                                               LAMP_MOVE_ELEVATION_MAX)
        lamp_angle_changed = sample_lamp_rotation(LAMP_MOVE_AZIMUTH_MIN,
                                                  LAMP_MOVE_AZIMUTH_MAX,
                                                  LAMP_MOVE_ELEVATION_MIN,
                                                  LAMP_MOVE_ELEVATION_MAX)

        # load scene with base obj
        pth = osp.join(sample_dir, "meshes", postures_df["base"]+".obj")
        verts_base, faces_base = umitsuba.load_obj(pth)
        renderer.replace_mesh(verts_base, faces_base)
        mooney1, _ = create_img(renderer, lamp_angle_base, spp)
        mooney2, _ = create_img(renderer, lamp_angle_changed, spp)
        pixel_distance_light = image_distance(mooney1, mooney2)

        # load scene with change obj
        pth = osp.join(sample_dir, "meshes", postures_df["change"]+".obj")
        verts_changed, faces_changed = umitsuba.load_obj(pth)
        renderer.replace_mesh(verts_changed, faces_changed)
        mooney2, _ = create_img(renderer, lamp_angle_base, spp)
        pixel_distance_pose = image_distance(mooney1, mooney2)

        dist_ratio = pixel_distance_light/pixel_distance_pose
        data.append({
            "scene": args.scene_id,
            "sample_id": n,
            "lamp_angle_base": lamp_angle_base,
            "lamp_angle_changed": lamp_angle_changed,
            "pixel_distance_light": pixel_distance_light,
            "pixel_distance_posture": pixel_distance_pose,
            "pixel_distance_ratio": dist_ratio,
        })
        if n % 20 == 0:
            pd.DataFrame.from_records(data).to_csv(df_pth, index=False)
            print(f"Saved info as {df_pth}.")

    pd.DataFrame.from_records(data).to_csv(df_pth, index=False)
    print(f"Saved info as {df_pth}.")

if __name__ == "__main__":
    main()
