import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_bp_data(manifest_path, plot_output_dir):
    """
    Reads the manifest CSV and generates BP plots for each subject.

    Args:
        manifest_path (str): The full path to the manifest.csv file.
        plot_output_dir (str): The directory where the plot images will be saved.
    """
    print("\n--- Generating Blood Pressure Plots ---")

    # Create the output directory for plots if it doesn't exist
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)
        print(f"Created directory for plots: {plot_output_dir}")

    try:
        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv(manifest_path)
        print("Successfully loaded manifest.csv.")
    except FileNotFoundError:
        print(
            f"Error: Manifest file not found at {manifest_path}. Cannot generate plots."
        )
        return

    # Ensure the BP columns are treated as numbers for plotting
    try:
        df["sbp"] = pd.to_numeric(df["sbp"])
        df["dbp"] = pd.to_numeric(df["dbp"])
        df["mbp"] = pd.to_numeric(df["mbp"])
    except Exception as e:
        print(f"Error converting BP data to numbers: {e}")
        return

    # Create a new column for the main subject identifier (e.g., 'Subject_01' from 'Subject_01_1')
    df["main_subject"] = df["subject_id"].str.split("_").str[:2].str.join("_")

    # Group the data by each unique 'main_subject'
    grouped = df.groupby("main_subject")

    # Loop through each subject's data and create a plot
    for main_subject_id, group in grouped:
        print(f"  - Plotting data for {main_subject_id}")

        # Set up the plot
        plt.figure(figsize=(15, 7))

        # Create an index for the x-axis (e.g., 1, 2, 3...) for each fragment
        fragment_index = range(1, len(group) + 1)

        # Plot each blood pressure component
        plt.plot(
            fragment_index,
            group["sbp"],
            marker="o",
            linestyle="-",
            label="SBP (Systolic)",
        )
        plt.plot(
            fragment_index,
            group["dbp"],
            marker="s",
            linestyle="-",
            label="DBP (Diastolic)",
        )
        plt.plot(
            fragment_index,
            group["mbp"],
            marker="^",
            linestyle="-",
            label="MBP (Mean Arterial)",
        )

        # Add titles and labels for clarity
        plt.title(f"Blood Pressure Readings for {main_subject_id}")
        plt.xlabel("Video Fragment Number (Across All Sessions)")
        plt.ylabel("Blood Pressure (mmHg)")
        plt.grid(True)
        plt.legend()

        # Define the filename and save the plot
        plot_filename = f"{main_subject_id}_bp_plot.png"
        plot_save_path = os.path.join(plot_output_dir, plot_filename)

        try:
            plt.savefig(plot_save_path)
            print(f"    - Saved plot to {plot_save_path}")
        except Exception as e:
            print(f"    - Error saving plot for {main_subject_id}: {e}")

        plt.close()  # Close the figure to free up memory before the next loop

    print(f"\n--- All plots have been saved to: {plot_output_dir} ---")


if __name__ == "__main__":
    # --- STEP 1: CONFIGURE YOUR PATHS HERE ---
    # The folder where your data and manifest.csv are located
    base_data_dir = r"C:\Users\yeonj\Desktop\iPPG_new_project\data"

    # The full path to your manifest file
    manifest_path = os.path.join(base_data_dir, "manifest.csv")

    # The folder where you want to save the generated graphs
    plot_dir = os.path.join(base_data_dir, "bp_plots")

    # --- STEP 2: RUN THE SCRIPT ---
    plot_bp_data(manifest_path, plot_dir)
