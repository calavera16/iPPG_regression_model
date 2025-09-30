import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_bp_data(manifest_path, plot_output_dir):
    """
    Reads the manifest CSV and generates a single plot for all subjects.

    Args:
        manifest_path (str): The full path to the manifest.csv file.
        plot_output_dir (str): The directory where the plot images will be saved.
    """
    print("\n--- Generating Combined Blood Pressure Plot ---")

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

    print("  - Plotting data for ALL subjects combined...")

    # Set up the plot
    plt.figure(figsize=(30, 15))

    # Create an index for the x-axis to represent each data point
    data_point_index = range(len(df))

    # Create scatter plots for each BP component to show all data points
    plt.scatter(
        data_point_index, df["sbp"], marker="o", alpha=0.6, s=10, label="SBP (Systolic)"
    )
    plt.scatter(
        data_point_index,
        df["dbp"],
        marker="s",
        alpha=0.6,
        s=10,
        label="DBP (Diastolic)",
    )
    plt.scatter(
        data_point_index,
        df["mbp"],
        marker="^",
        alpha=0.6,
        s=10,
        label="MBP (Mean Arterial)",
    )

    # Add titles and labels for clarity
    plt.title("Blood Pressure Readings (All Subjects)")
    plt.xlabel("Data Point Index")
    plt.ylabel("Blood Pressure (mmHg)")
    plt.grid(True)
    plt.legend()

    # Define the filename and save the single plot
    plot_filename = "all_subjects_bp_plot.png"
    plot_save_path = os.path.join(plot_output_dir, plot_filename)

    try:
        plt.savefig(plot_save_path)
        print(f"    - Saved combined plot to {plot_save_path}")
    except Exception as e:
        print(f"    - Error saving combined plot: {e}")

    plt.close()  # Close the figure

    print(f"\n--- Combined plot has been saved to: {plot_output_dir} ---")
    print(
        f"SBP mean : {df['sbp'].mean():.2f}, DBP mean : {df['dbp'].mean():.2f}, MBP mean : {df['mbp'].mean():.2f}"
    )
    print(
        f"SBP std : {df['sbp'].std():.2f}, DBP std : {df['dbp'].std():.2f}, MBP std : {df['mbp'].std():.2f}"
    )
    print(
        f"SBP max : {df['sbp'].max():.2f}, DBP max : {df['dbp'].max():.2f}, MBP max : {df['mbp'].max():.2f}"
    )
    print(
        f"SBP min : {df['sbp'].min():.2f}, DBP min : {df['dbp'].min():.2f}, MBP min : {df['mbp'].min():.2f}"
    )
    print(f"the number of dbp under 40 is: {(df['dbp'] < 40).sum()}")
    print(
        f"the percentage of dbp under 40 is: {(df['dbp'] < 40).sum() / len(df) * 100:.2f}%"
    )
    print(f"the number of dbp over 80 is: {(df['dbp'] > 80).sum()}")
    print(
        f"the percentage of dbp under 40 is: {(df['dbp'] > 80).sum() / len(df) * 100:.2f}%"
    )


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
