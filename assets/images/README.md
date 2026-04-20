# Example Image Resources

These directories contain example assets referenced by the README and the bundled workflows.

- `deg_crop/`: degraded aligned face crops
- `ref_crop/`: aligned reference face crops
- `deg_whole/`: degraded full-image examples
- `ref_whole/`: full-image reference examples

Typical usage:

1. Copy the images you want to test into `ComfyUI/input/`.
2. Load `workflows/aligned_face_restore.json` if you already have aligned face crops.
3. Load `workflows/full_image_restore.json` if you want to start from original full images and use RetinaFace detection + align + paste-back.

The screenshots in the top-level `assets/` folder illustrate both workflow types:

- `../aligned_restoration.png`
- `../whole_restoration.png`
