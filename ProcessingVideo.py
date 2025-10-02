import os
import cv2
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# --- 1. Configuration ---
# Point this to your ORIGINAL manifest file
ORIGINAL_MANIFEST_FILE = (
    r"C:\Users\yeonj\Desktop\iPPG_new_project\data\manifest copy.csv"
)

# Define where to save the pre-processed tensors and the new manifest
PREPROCESSED_DATA_DIR = r"C:\Users\yeonj\Desktop\iPPG_new_project\256by256npy"
NEW_MANIFEST_FILE = os.path.join(
    os.path.dirname(PREPROCESSED_DATA_DIR), "manifest_preprocessed_copy_256by256.csv"
)

# These parameters must match the ones in your training script
NUM_FRAMES = 150
RESIZE_DIM = (256, 256)


# --- 2. Main Pre-processing Logic ---
def preprocess_videos():
    """
    Reads the original manifest, processes each video into a tensor,
    saves the tensor to disk, and creates a new manifest file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

    manifest_df = pd.read_csv(ORIGINAL_MANIFEST_FILE)
    new_manifest_data = []

    print(f"Starting pre-processing of {len(manifest_df)} videos...")
    print(f"Processed tensors will be saved in: {PREPROCESSED_DATA_DIR}")

    # Use tqdm for a progress bar
    for idx, row in tqdm(
        manifest_df.iterrows(), total=len(manifest_df), desc="Processing Videos"
    ):
        video_path = row["fragment_path"]

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video file {video_path}. Skipping.")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frames = []
            indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (RESIZE_DIM[1], RESIZE_DIM[0]))
                    frames.append(frame)
                else:
                    # If a frame can't be read, duplicate the last good one
                    if len(frames) > 0:
                        frames.append(frames[-1])
                    else:
                        # If no frames can be read, this will create a black video tensor
                        # and allow the loop to continue.
                        frames.append(
                            np.zeros((RESIZE_DIM[0], RESIZE_DIM[1], 3), dtype=np.uint8)
                        )

            cap.release()

            # Convert to tensor and permute dimensions
            video_tensor = torch.from_numpy(np.array(frames, dtype=np.float32)) / 255.0
            video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, D, H, W)

            # Define path for the new .pt file
            base_filename = os.path.basename(video_path)
            tensor_filename = os.path.splitext(base_filename)[0] + ".pt"
            tensor_save_path = os.path.join(PREPROCESSED_DATA_DIR, tensor_filename)

            # Save the tensor
            torch.save(video_tensor, tensor_save_path)

            # Add data to the new manifest list
            new_row = row.to_dict()
            new_row["tensor_path"] = tensor_save_path  # Add the new path
            new_manifest_data.append(new_row)

        except Exception as e:
            print(f"Error processing {video_path}: {e}. Skipping.")

    # Create and save the new manifest DataFrame
    new_manifest_df = pd.DataFrame(new_manifest_data)
    new_manifest_df.to_csv(NEW_MANIFEST_FILE, index=False)

    print("\nPre-processing complete!")
    print(f"New manifest file saved to: {NEW_MANIFEST_FILE}")


if __name__ == "__main__":
    preprocess_videos()
