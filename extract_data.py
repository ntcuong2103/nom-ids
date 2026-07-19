import glob
dataset = "datasets/nomnaocr/Pages"
output_dir = "datasets/nomnaocr/images"
import os
import shutil


def main():
    # Save all page images to the training image folder.
    os.makedirs(output_dir, exist_ok=True)
    for img in glob.glob(f"{dataset}/**/*.jpg", recursive=True):
        id = os.path.basename(img)[:-4]
        folder = os.path.basename(os.path.dirname(os.path.dirname(img)))
        os.makedirs(f"{output_dir}/{folder}", exist_ok=True)
        shutil.copy(img, f"{output_dir}/{folder}/{id}.jpg")


if __name__ == "__main__":
    main()
