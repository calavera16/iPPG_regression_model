import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

# Import DALI libraries
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIDataLoader

# --- 1. Configuration ---
MANIFEST_FILE = r"C:\Users\yeonj\Desktop\iPPG_new_project\data\manifest copy.csv"
RESULTS_DIR = r"C:\Users\yeonj\Desktop\iPPG_new_project\results_model2_2"
NUM_EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = "cuda"
NUM_FOLDS = 5
DROPOUT_RATE = 0.5
EARLY_STOPPING_PATIENCE = 10

# Video processing parameters for DALI
NUM_FRAMES = 90
NUM_CLIPS = 3
CLIP_LEN = NUM_FRAMES // NUM_CLIPS
RESIZE_DIM = (128, 128)

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")
print(f"Starting {NUM_FOLDS}-fold cross-validation with NVIDIA DALI.")


# --- 2. DALI Video Pipeline ---
# This class defines the data loading and processing that will happen ON THE GPU.
def DALIVideoPipeline(manifest_df, sequence_length, random_shuffle=True):
    video_files = manifest_df["fragment_path"].tolist()
    labels = manifest_df[["sbp", "dbp"]].values.astype(np.float32)

    pipeline = Pipeline(batch_size=BATCH_SIZE, num_threads=4, device_id=0)
    with pipeline:
        videos, dali_labels = fn.readers.video_resize(
            device="gpu",
            filenames=video_files,
            labels=labels,
            sequence_length=sequence_length,
            normalized=False,  # We will normalize with fn.crop_mirror_normalize
            random_shuffle=random_shuffle,
            image_type=types.RGB,
            dtype=types.UINT8,  # Read as uint8 before converting to float
            resize_x=RESIZE_DIM[1],
            resize_y=RESIZE_DIM[0],
            name="Reader",
        )
        # Normalize and convert to float
        videos = fn.crop_mirror_normalize(
            videos, dtype=types.FLOAT, mean=[0.0], std=[255.0]
        )
        videos = fn.transpose(videos, perm=[3, 0, 1, 2])
        pipeline.set_outputs(videos, dali_labels)
    return pipeline


# --- 3. Model Architecture (No changes needed) ---
class VideoBPNet(nn.Module):
    def __init__(self, num_clips, clip_len, dropout_rate):
        super(VideoBPNet, self).__init__()
        self.num_clips = num_clips
        self.clip_len = clip_len

        self.cnn = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, 2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size * self.num_clips, 3, self.clip_len, *RESIZE_DIM)
        features = self.cnn(x)
        features = features.view(batch_size, self.num_clips, -1)
        aggregated_features = torch.mean(features, dim=1)
        out = self.fc(aggregated_features)
        return out


# --- 4. Plotting and History Functions ---
def plot_and_save_history(history, fold, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(f"Fold {fold} Training History", fontsize=16)
    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss (MSE)")
    ax1.plot(history["epoch"], history["test_loss"], label="Test Loss (MSE)")
    (
        ax1.set_ylabel("Loss (MSE)"),
        ax1.set_title("Mean Squared Error Loss vs. Epochs"),
        ax1.legend(),
        ax1.grid(True),
    )
    (
        ax2.plot(history["epoch"], history["train_mae"], label="Train MAE"),
        ax2.plot(history["epoch"], history["test_mae"], label="Test MAE"),
    )
    (
        ax2.set_xlabel("Epoch"),
        ax2.set_ylabel("Mean Absolute Error (MAE)"),
        ax2.set_title("Mean Absolute Error vs. Epochs"),
        ax2.legend(),
        ax2.grid(True),
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]), plt.savefig(save_path), plt.close()


def plot_subject_metrics(subject_metrics_df, fold, save_path):
    subject_metrics_df = subject_metrics_df.sort_values(by="subject_id")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    fig.suptitle(f"Fold {fold} - Per-Subject Metrics", fontsize=16)
    subjects = subject_metrics_df["subject_id"]
    (
        ax1.bar(subjects, subject_metrics_df["mse"]),
        ax1.set_ylabel("MSE Loss"),
        ax1.set_title("Mean Squared Error per Subject"),
        ax1.tick_params(axis="x", rotation=45),
        ax1.grid(axis="y"),
    )
    (
        ax2.bar(subjects, subject_metrics_df["mae"]),
        ax2.set_ylabel("MAE"),
        ax2.set_title("Mean Absolute Error per Subject"),
        ax2.tick_params(axis="x", rotation=45),
        ax2.grid(axis="y"),
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]), plt.savefig(save_path), plt.close()


# --- 5. Training and Evaluation Functions (Completed for DALI loader) ---
def train_and_test(
    model,
    train_loader,
    test_loader,
    optimizer,
    mse_criterion,
    mae_criterion,
    epochs,
    writer,
    fold_best_model_path,
    scaler,
):
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_mae": [],
        "test_mae": [],
    }
    best_test_mae = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_mae = 0.0, 0.0
        train_samples = 0
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False
        )
        for data in train_pbar:
            videos = data[0]["videos"]
            labels = data[0]["labels"]
            train_samples += videos.size(0)
            with torch.cuda.amp.autocast():
                outputs = model(videos)
                loss = mse_criterion(outputs, labels)
            (
                optimizer.zero_grad(),
                scaler.scale(loss).backward(),
                scaler.step(optimizer),
                scaler.update(),
            )
            train_loss += loss.item() * videos.size(0)
            with torch.no_grad():
                mae = mae_criterion(outputs, labels)
                train_mae += mae.item() * videos.size(0)
            train_pbar.set_postfix({"MSE": loss.item(), "MAE": mae.item()})
        train_loader.reset()
        train_loss /= train_samples
        train_mae /= train_samples

        model.eval()
        test_loss, test_mae = 0.0, 0.0
        test_samples = 0
        with torch.no_grad():
            test_pbar = tqdm(
                test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Test]", leave=False
            )
            for data in test_pbar:
                videos = data[0]["videos"]
                labels = data[0]["labels"]
                test_samples += videos.size(0)
                with torch.cuda.amp.autocast():
                    outputs = model(videos)
                    loss = mse_criterion(outputs, labels)
                    mae = mae_criterion(outputs, labels)
                test_loss += loss.item() * videos.size(0)
                test_mae += mae.item() * videos.size(0)
                test_pbar.set_postfix({"MSE": loss.item(), "MAE": mae.item()})
        test_loader.reset()
        test_loss /= test_samples
        test_mae /= test_samples

        print(
            f"Epoch {epoch + 1}/{epochs} Summary | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}"
        )

        if test_mae < best_test_mae:
            best_test_mae = test_mae
            torch.save(model.state_dict(), fold_best_model_path)
            print(
                f"  ---> New best model for this fold saved to {os.path.basename(fold_best_model_path)} (MAE: {best_test_mae:.4f})"
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        (
            writer.add_scalar("Loss/Train_MSE", train_loss, epoch + 1),
            writer.add_scalar("Loss/Test_MSE", test_loss, epoch + 1),
        )
        (
            writer.add_scalar("MAE/Train", train_mae, epoch + 1),
            writer.add_scalar("MAE/Test", test_mae, epoch + 1),
        )
        (
            history["epoch"].append(epoch + 1),
            history["train_loss"].append(train_loss),
            history["test_loss"].append(test_loss),
        )
        history["train_mae"].append(train_mae), history["test_mae"].append(test_mae)

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1} after {EARLY_STOPPING_PATIENCE} epochs with no improvement."
            )
            break

    return history


def evaluate_and_save_predictions(model, test_manifest, writer, fold):
    model.eval()
    all_preds, all_labels = [], []

    # Create a non-shuffled DALI pipeline for evaluation
    eval_pipe = DALIVideoPipeline(test_manifest, NUM_FRAMES, random_shuffle=False)
    eval_loader = DALIDataLoader(
        eval_pipe, output_map=["videos", "labels"], last_batch_policy="PARTIAL"
    )

    with torch.no_grad():
        for data in tqdm(eval_loader, desc="Generating Final Predictions", leave=False):
            videos = data[0]["videos"]
            labels = data[0]["labels"]
            with torch.cuda.amp.autocast():
                outputs = model(videos)
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
    eval_loader.reset()

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds_df = pd.DataFrame(
        {
            "actual_sbp": all_labels[:, 0],
            "predicted_sbp": all_preds[:, 0],
            "actual_dbp": all_labels[:, 1],
            "predicted_dbp": all_preds[:, 1],
        }
    )
    preds_df.to_csv(
        os.path.join(RESULTS_DIR, f"fold_{fold}_predictions.csv"), index=False
    )
    print(f"Saved predictions for fold {fold} to 'fold_{fold}_predictions.csv'")

    mae_sbp = torch.abs(all_labels[:, 0] - all_preds[:, 0]).mean().item()
    mae_dbp = torch.abs(all_labels[:, 1] - all_preds[:, 1]).mean().item()
    mse_sbp = torch.pow(all_labels[:, 0] - all_preds[:, 0], 2).mean().item()
    mse_dbp = torch.pow(all_labels[:, 1] - all_preds[:, 1], 2).mean().item()

    print(f"\n--- Fold {fold} Detailed Test Metrics ---")
    print(f"  SBP MAE : {mae_sbp:.4f}"), print(f"  DBP MAE : {mae_dbp:.4f}")
    print(f"  SBP MSE : {mse_sbp:.4f}"), print(f"  DBP MSE : {mse_dbp:.4f}")

    writer.add_hparams(
        {"fold": fold},
        {
            "final_mae_sbp": mae_sbp,
            "final_mae_dbp": mae_dbp,
            "final_mse_sbp": mse_sbp,
            "final_mse_dbp": mse_dbp,
        },
    )
    return {
        "mae_sbp": mae_sbp,
        "mae_dbp": mae_dbp,
        "mse_sbp": mse_sbp,
        "mse_dbp": mse_dbp,
    }


# --- 6. Main Execution with K-Fold Cross-Validation ---
if __name__ == "__main__":
    print("\n--- Model Architecture ---")
    temp_model = VideoBPNet(
        num_clips=NUM_CLIPS, clip_len=CLIP_LEN, dropout_rate=DROPOUT_RATE
    )
    print(temp_model)
    with open(os.path.join(RESULTS_DIR, "model_architecture.txt"), "w") as f:
        f.write(str(temp_model))
    print("Model architecture saved to results/model_architecture.txt")
    del temp_model

    manifest_df = pd.read_csv(MANIFEST_FILE)
    subjects = manifest_df["subject_id"].unique()
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_subject_indices, test_subject_indices) in enumerate(
        kf.split(subjects)
    ):
        print(f"\n{'=' * 25} FOLD {fold + 1}/{NUM_FOLDS} {'=' * 25}")

        train_subjects, test_subjects = (
            subjects[train_subject_indices],
            subjects[test_subject_indices],
        )
        train_manifest_df = manifest_df[manifest_df["subject_id"].isin(train_subjects)]
        test_manifest_df = manifest_df[manifest_df["subject_id"].isin(test_subjects)]

        print(
            f"Training on {len(train_subjects)} subjects ({len(train_manifest_df)} videos)."
        )
        print(
            f"Testing on {len(test_subjects)} subjects ({len(test_manifest_df)} videos)."
        )

        # Create DALI pipelines for this fold
        train_pipe = DALIVideoPipeline(
            train_manifest_df, NUM_FRAMES, random_shuffle=True
        )
        test_pipe = DALIVideoPipeline(
            test_manifest_df, NUM_FRAMES, random_shuffle=False
        )

        train_loader = DALIDataLoader(
            train_pipe,
            output_map=["videos", "labels"],
            last_batch_policy="PARTIAL",
            reader_name="Reader",
        )
        test_loader = DALIDataLoader(
            test_pipe,
            output_map=["videos", "labels"],
            last_batch_policy="PARTIAL",
            reader_name="Reader",
        )

        print("Initializing a new model and TensorBoard writer for this fold...")
        model = VideoBPNet(
            num_clips=NUM_CLIPS, clip_len=CLIP_LEN, dropout_rate=DROPOUT_RATE
        ).to(DEVICE)
        writer = SummaryWriter(
            log_dir=os.path.join(RESULTS_DIR, "runs", f"fold_{fold + 1}")
        )

        mse_criterion, mae_criterion = nn.MSELoss(), nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()

        fold_best_model_path = os.path.join(
            RESULTS_DIR, f"best_model_fold_{fold + 1}.pth"
        )

        history = train_and_test(
            model,
            train_loader,
            test_loader,
            optimizer,
            mse_criterion,
            mae_criterion,
            epochs=NUM_EPOCHS,
            writer=writer,
            fold_best_model_path=fold_best_model_path,
            scaler=scaler,
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(
            os.path.join(RESULTS_DIR, f"fold_{fold + 1}_history.csv"), index=False
        )
        plot_and_save_history(
            history, fold + 1, os.path.join(RESULTS_DIR, f"fold_{fold + 1}_metrics.png")
        )
        print(f"Saved epoch history and plot for fold {fold + 1}.")

        print(
            f"Loading best model from {os.path.basename(fold_best_model_path)} for final evaluation."
        )
        model.load_state_dict(torch.load(fold_best_model_path))

        detailed_metrics = evaluate_and_save_predictions(
            model, test_manifest_df, writer, fold + 1
        )
        fold_results.append(
            {
                "fold": fold + 1,
                "test_loss_mse": history["test_loss"][-1],
                "test_mae": history["test_mae"][-1],
                **detailed_metrics,
            }
        )
        writer.close()

    print(f"\n{'=' * 20} CROSS-VALIDATION SUMMARY {'=' * 20}")
    results_df = pd.DataFrame(fold_results)

    for metric in results_df.columns:
        if metric != "fold":
            avg = results_df[metric].mean()
            std = results_df[metric].std()
            print(
                f"Average Test {metric.upper()} across {NUM_FOLDS} folds: {avg:.4f} (std: {std:.4f})"
            )

    print("\nIndividual fold results:")
    print(results_df.to_string(index=False))
