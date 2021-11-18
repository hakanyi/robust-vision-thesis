## Stimulus generation pipeline
1. Manually choose base and change postures from H36M dataset. The outcome of this is a `base_posture.txt` file that we'll put in a folder, e.g. `sampled-lights`.
2. Use `00_h36m_to_mesh.py` to generate meshes for all of the images from `base_postures.txt` and place them in the `meshes` folder under `sampled-lights`.
3. Use `01_sample_candidates.sh` to go through the mesh pairs and, for each, generate N base:changed-light and base:changed-pose pairs where the underlying lamp position is sampled. Record image statistics under `sampled-lights`, but don't save images.
4. Analyze the data with `02_analyze_candidates.Rmd` to determine a) pairs where the pixel distance due to light changes are comparable to pixel distance due to posture changes and b) out these pairs, whether the pixel distances lie in a given range. Save the filtered csv to `candidate_pairs.csv`.
5. Produce the images in `candidate_pairs.csv` (incl. diff-images) to sanity-check using `03_visualize_candidates.py`. Place them in `candidate_pairs_images`.
6. Across all `sampled-lights-<x>` folders, choose from the candidates and consolidate the output in a `image_info.csv` in this folder.
7. Consolidate the images in `candidate_pairs.csv` and place them in `images` using `04_collect_images.py`.
8. Use `05_make_videos.py` to produce the video stimuli for the behavioral experiment.
