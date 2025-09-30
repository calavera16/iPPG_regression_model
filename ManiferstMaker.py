import os
import csv
import re


def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    """
    Provides a key for sorting strings in a natural order (e.g., 'item2' before 'item10').
    """
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def parse_bp_data(txt_path):
    """
    Parses a BP text file to extract SBP, MAP, and DBP readings.

    Args:
        txt_path (str): The path to the BP data text file.

    Returns:
        list: A list of tuples, where each tuple contains (SBP, MAP, DBP).
              Returns an empty list if the file can't be read.
    """
    bp_readings = []
    try:
        with open(txt_path, "r") as f:
            # Skip the header line
            next(f)
            for line in f:
                parts = line.split()
                if len(parts) >= 7:
                    # NIBP SYS is column 5, MAP is 6, DIA is 7
                    sbp = parts[5]
                    mbp = parts[6]
                    dbp = parts[7]
                    bp_readings.append((sbp, mbp, dbp))
    except FileNotFoundError:
        print(f"  - Warning: BP file not found at {txt_path}")
        return []
    except Exception as e:
        print(f"  - Warning: Error reading BP file {txt_path}: {e}")
        return []
    return bp_readings


def create_manifest(fragment_videos_dir, bp_data_dir, manifest_output_path):
    """
    Creates a CSV manifest mapping video fragments to BP readings.
    """
    print("\n--- Creating Manifest File ---")
    manifest_data = []

    # Get all subject directories (e.g., 'Subject_01_1')
    subject_dirs = [
        d
        for d in os.listdir(fragment_videos_dir)
        if os.path.isdir(os.path.join(fragment_videos_dir, d))
    ]

    for subject_id in subject_dirs:
        print(f"  - Processing subject: {subject_id}")

        # Transform the subject_id to match the BP filename format
        # e.g., 'Subject_01_1' -> 'Subject_01_nibp_1.txt'
        parts = subject_id.split("_")
        if len(parts) == 3:  # Handles 'Subject_01_1'
            base_name = f"{parts[0]}_{parts[1]}"  # Creates 'Subject_01'
            session_num = parts[2]  # Gets '1'
            bp_txt_filename = f"{base_name}_nibp_{session_num}.txt"
        else:
            # Fallback for any unexpected folder names
            print(
                f"  - Warning: Could not parse subject ID '{subject_id}'. Using default naming."
            )
            bp_txt_filename = f"{subject_id}.txt"

        bp_txt_path = os.path.join(bp_data_dir, bp_txt_filename)

        bp_readings = parse_bp_data(bp_txt_path)

        # If there are no BP readings at all, skip this subject.
        if not bp_readings:
            print(
                f"  - Warning: Skipping subject {subject_id} due to NO BP data found."
            )
            continue

        # Check if there are fewer than 3 readings and print an info message.
        has_enough_readings = len(bp_readings) >= 3
        if not has_enough_readings:
            print(
                f"  - Info: Subject {subject_id} has fewer than 3 BP readings. Using the first reading for all fragments."
            )

        # Get all video fragments for this subject and sort them naturally
        subject_video_dir = os.path.join(fragment_videos_dir, subject_id)
        video_fragments = [
            f
            for f in os.listdir(subject_video_dir)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]
        video_fragments.sort(key=natural_sort_key)

        for i, fragment_name in enumerate(video_fragments):
            fragment_number = i + 1  # Use 1-based index for logic
            fragment_path = os.path.join(subject_video_dir, fragment_name)

            sbp, mbp, dbp = None, None, None

            # If there are enough readings, use the interval-based logic.
            if has_enough_readings:
                if 1 <= fragment_number <= 20:
                    sbp, mbp, dbp = bp_readings[0]
                elif 21 <= fragment_number <= 40:
                    sbp, mbp, dbp = bp_readings[1]
                elif 41 <= fragment_number <= 60:
                    sbp, mbp, dbp = bp_readings[2]
                else:
                    # If there are more than 60 fragments, skip them
                    continue
            # Otherwise, use the first available reading for all fragments (up to 60).
            else:
                if 1 <= fragment_number <= 60:
                    sbp, mbp, dbp = bp_readings[0]
                else:
                    continue

            # Add the data to the manifest if a BP reading was assigned.
            if sbp is not None:
                manifest_data.append([fragment_path, subject_id, sbp, dbp, mbp])

    # Write the collected data to the CSV file
    try:
        with open(manifest_output_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["fragment_path", "subject_id", "sbp", "dbp", "mbp"])
            # Write data rows
            writer.writerows(manifest_data)
        print(
            f"\n--- Manifest file created successfully at: {manifest_output_path} ---"
        )
    except Exception as e:
        print(f"\n--- Error writing manifest file: {e} ---")


if __name__ == "__main__":
    # --- STEP 1: CONFIGURE YOUR PATHS HERE ---
    # This script creates a manifest file from existing video fragments and BP data.

    # Path to the folder containing your subject sub-folders with the pre-split video fragments
    fragment_videos_dir = (
        r"C:\Users\yeonj\Desktop\iPPG_new_project\data\Videos_processed"
    )

    # Path to the folder containing your BP .txt files
    # Make sure your .txt files are named to match the subject folders (e.g., 'Subject_01_1.txt')
    bp_data_dir = r"C:\Users\yeonj\Desktop\iPPG_new_project\data\NIBP_data_processed"

    # Full path for the final output CSV file
    manifest_path = r"C:\Users\yeonj\Desktop\iPPG_new_project\data\manifest.csv"

    # --- STEP 2: RUN THE SCRIPT ---
    create_manifest(fragment_videos_dir, bp_data_dir, manifest_path)
