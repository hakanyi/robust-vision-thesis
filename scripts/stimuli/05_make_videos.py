import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tqdm import tqdm

import robust_vision.utils.workspace as ws

def get_arguments():
    arg_parser = argparse.ArgumentParser(
        description="Make change detection videos in 2x2 grids."
    )
    arg_parser.add_argument(
        "--exp-dir", "-e", dest="exp_dir", required=True,
        help="Experiment directory.",
    )
    arg_parser.add_argument(
        "--stimulus-time", "-st", dest="stimulus_time",
        type=int, default=250,
        help="Stimulus presentation time in ms.",
    )
    arg_parser.add_argument(
        "--mask-time", "-mt", dest="mask_time",
        type=int, default=250,
        help="Mask presentation time in ms.",
    )
    arg_parser.add_argument(
        "--save-dir-suffix", "-sds", dest="save_dir_suffix",
        type=str, default=None,
        help="Name suffix for the directory to store output. If not \
        provided, current time will be used."
    )
    arg_parser.add_argument(
        "--upside-down", "-usd", dest="upside_down", action="store_true",
        help = "Whether to vertically flip images before making the grids."
    )
    arg_parser.add_argument(
        "--overwrite", "-o", dest="overwrite", action="store_true",
        help = "Whether to overwrite the experiment folder."
    )

    args = arg_parser.parse_args()
    return args

def random_grid_coords():
    return np.random.choice([0,1,2,3], 2, replace=False)

def make_movie(out_pth, img1_pth, img2_pth, grid_coords,
               stim_time, mask_time, fps=60, upside_down=False):
    w, h = Image.open(img1_pth).size
    assert (w, h) == Image.open(img2_pth).size
    flip = "vflip" if upside_down else "null"
    # grid coordinates |0,0|0,1|
    #                  |1,0|1,1|
    x1, y1 = np.array(np.unravel_index(grid_coords[0], (2,2))) * np.array([w, h])
    x2, y2 = np.array(np.unravel_index(grid_coords[1], (2,2))) * np.array([w, h])
    st = stim_time/1000.; mt = mask_time/1000.
    complex_filter = [
        "\"",
        # images, optionally flipped
        f"[1:v]{flip}[i1];", f"[3:v]{flip}[i2];",
        # grid 1: stimulus
        f"[0:v][i1]overlay={x1}:{y1}:enable='between(t,0,{st})'[g1];",
        # grid 2: mask
        f"[g1][2:v]overlay=0:0:enable='between(t,{st},{st+mt})'[g2];",
        # grid 2: background and, at the same time, ...
        f"[g2][0:v]overlay=0:0:enable='between(t,{st+mt},{2*st+mt})'[g3];"
        # grid 2: ... fill quadrant with stimulus
        f"[g3][i2]overlay={x2}:{y2}:enable='between(t,{st+mt},{2*st+mt})'",
        "\""
    ]
    cmd = [
        "ffmpeg -y -hide_banner -loglevel error",
        f"-f lavfi -i color=black:{w*2}x{h*2}", # background 0
        f"-i {img1_pth}", # img1 1
        f"-f lavfi -i color=white:{w*2}x{h*2}", # mask 2
        f"-i {img2_pth}", # img2 3
        f"""-filter_complex""", ''.join(complex_filter), # filter
        f"-pix_fmt yuv420p -r {fps} -t {2*st+mt} {out_pth}"
    ]
    cmd = " ".join(cmd)
    os.system(cmd)

def info_dict(gr_truth, mode):
    keys_keep = ["light-distance", "mesh-distance",
                 "posevec-distance", "shapevec-distance"]
    grt = {key: value for key, value in gr_truth.items() if key in keys_keep}
    if mode == "same-image":
        # all distances are 0
        grt = dict.fromkeys(grt, 0)
        grt['pixel-distance'] = 0
    elif mode == "different-light":
        grt["mesh-distance"] = 0
        grt["posevec-distance"] = 0
        grt["shapevec-distance"] = 0
        grt["pixel-distance"] = gr_truth["pixel_distance-different_light"]
    elif mode == "different-posture":
        grt["light-distance"] = 0
        grt["pixel-distance"] = gr_truth["pixel_distance-different_pose"]

    return grt

def make_movies(img_dir, movie_dir, pair_id, gr_truth,
                stim_time, mask_time, upside_down):

    def get_pth(suffix):
        return osp.join(img_dir, f"{pair_id:02d}_{suffix}_mooney.png")
    img_base = get_pth("base")
    img_difflight = get_pth("light")
    img_diffpose = get_pth("posture")

    def make_name(key):
        return f"{pair_id:02d}_{key}.mp4"

    def make_mode(key, img1_pth, img2_pth):
        out_name = make_name(key)
        out_pth = osp.join(movie_dir, out_name)
        make_movie(out_pth, img1_pth, img2_pth, random_grid_coords(),
                   stim_time, mask_time, upside_down=upside_down)
        info = info_dict(gr_truth, key)
        info["name"] = out_name
        info["stim_time"] = stim_time/1000.
        info["mask_time"] = mask_time/1000.
        return info

    info = []
    # base - different light
    info.append(make_mode("different-light", img_base, img_difflight))
    # base - different pose
    info.append(make_mode("different-posture", img_base, img_diffpose))
    # base - base
    info.append(make_mode("same-image", img_base, img_base))

    return info

if __name__ == "__main__":
    args = get_arguments()

    np.random.seed(0)

    # directory where we'll save the stimuli
    if args.save_dir_suffix is None:
        save_dir_suffix = f"_{datetime.now().strftime('%m-%d-%H:%M:%S')}"
    else:
        save_dir_suffix = f"_{args.save_dir_suffix}"

    stim_dir = osp.join(args.exp_dir, ws.STIMULI_DIR)

    img_dir = osp.join(stim_dir,"images")
    if not osp.exists(img_dir):
        raise Exception(f"Image directory {img_dir} does not exist.")

    movie_dir = os.path.join(stim_dir, f"movies{save_dir_suffix}")
    if osp.exists(movie_dir) and not args.overwrite:
        raise Exception(f"""Movie directory {movie_dir} exists.
        Use -o flag to overwrite.""")
    os.makedirs(movie_dir, exist_ok=args.overwrite)

    # read image info
    pth = osp.join(args.exp_dir, ws.STIMULI_DIR, "images", ws.IMG_OUT_FILE)
    stim_info_df = pd.read_csv(pth)
    stim_info_df.pair_id = stim_info_df.pair_id.astype(int)
    max_id = stim_info_df.pair_id.values.max()

    meta_data = []
    for pair_id in tqdm(stim_info_df.pair_id.unique()):
        gr_truth = stim_info_df[stim_info_df.pair_id == pair_id]
        gr_truth = gr_truth.to_dict(orient="records")[0]
        info = make_movies(img_dir, movie_dir, pair_id, gr_truth,
                           stim_time=args.stimulus_time,
                           mask_time=args.mask_time,
                           upside_down=args.upside_down)
        meta_data.extend(info)

    df = pd.DataFrame.from_records(meta_data)
    pth = osp.join(movie_dir, ws.VID_OUT_FILE)
    df.to_csv(pth, index=False)
    print(f"Saved metadata to {pth}.")
