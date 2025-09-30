import os
from moviepy.editor import VideoFileClip
import math


def split_video_into_fragments(video_path, output_directory, fragment_duration=10):
    """
    Splits a video into smaller fragments of a specified duration.

    Args:
        video_path (str): The full path to the input video file.
        output_directory (str): The directory where the fragments will be saved.
        fragment_duration (int): The duration of each fragment in seconds.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    try:
        print(f"\n--- Processing video: {os.path.basename(video_path)} ---")
        # Load the video clip
        clip = VideoFileClip(video_path)
        duration = clip.duration

        # Get the base filename and extension from the input path
        base_filename, file_extension = os.path.splitext(os.path.basename(video_path))

        print(f"  - Detected video duration: {duration:.2f} seconds.")

        # Calculate the number of fragments to create
        num_fragments = math.ceil(duration / fragment_duration)
        print(f"  - This will be split into {num_fragments} fragments.")

        # Loop through the video and create subclips
        for i in range(num_fragments):
            start_time = i * fragment_duration
            end_time = min((i + 1) * fragment_duration, duration)

            # This check prevents creating a zero-length clip at the very end
            if start_time >= end_time:
                continue

            # Define the output filename based on the desired format
            # e.g., Subject_01_1 -> Subject_01_1_1.mp4, Subject_01_1_2.mp4, etc.
            fragment_filename = f"{base_filename}_{i + 1}{file_extension}"
            fragment_path = os.path.join(output_directory, fragment_filename)

            print(
                f"  - Creating fragment {i + 1}/{num_fragments}: {fragment_filename} ({start_time:.2f}s to {end_time:.2f}s)"
            )

            # Create the subclip
            subclip = clip.subclip(start_time, end_time)

            # Write the subclip to the output file
            # You can customize codecs and other parameters here if needed
            subclip.write_videofile(
                fragment_path,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None,
            )

            # We are not explicitly closing the subclip here to let garbage collection handle it,
            # which can sometimes be more stable in a loop.

        # Close the main clip to release the file
        clip.close()
        print("--- Finished processing video ---")

    except Exception as e:
        print(f"An error occurred while processing {video_path}: {e}")


if __name__ == "__main__":
    print("--- Video Directory Splitter ---")
    print(
        "This script will find all videos in a folder and split them into 10-second fragments."
    )

    # --- SET YOUR PATHS HERE ---
    # The 'r' before the string is important for Windows paths.
    input_dir = r"C:\Users\yeonj\Desktop\iPPG_new_project\data\Videos"
    output_dir = r"C:\Users\yeonj\Desktop\iPPG_new_project\data\Videos_processed"

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Supported video file extensions
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv"]

    # Check if the input directory exists
    if os.path.isdir(input_dir):
        # Loop through all files in the input directory
        for filename in os.listdir(input_dir):
            # Check if the file has a supported video extension
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in video_extensions:
                video_path = os.path.join(input_dir, filename)

                # Create a specific sub-directory for each video's fragments to keep them organized
                video_base_name = os.path.splitext(filename)[0]
                video_output_folder = os.path.join(output_dir, video_base_name)

                split_video_into_fragments(video_path, video_output_folder)

        print("\nAll videos have been processed!")
    else:
        print(f"Error: The directory '{input_dir}' was not found.")
