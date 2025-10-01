import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

# --- 1. Configuration ---
MANIFEST_FILE = r"C:\Users\yeonj\Desktop\iPPG_new_project\data\manifest copy.csv"  # IMPORTANT: Update this path
RESULTS_DIR = r"C:\Users\yeonj\Desktop\iPPG_new_project\data\results_model_1"  # Directory to save architecture, plots, and history
NUM_EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FOLDS = 5

# Video processing parameters
NUM_FRAMES = 150
NUM_CLIPS = 5
CLIP_LEN = NUM_FRAMES // NUM_CLIPS
RESIZE_DIM = (225, 225)

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Starting {NUM_FOLDS}-fold cross-validation.")
print(f"All results will be saved in the '{RESULTS_DIR}' directory.")
print(
    f"To view live training metrics, run: tensorboard --logdir={os.path.join(RESULTS_DIR, 'runs')}"
)


# --- 2. Custom Dataset ---
class VideoBPDataset(Dataset):
    def __init__(self, manifest_df, resize_dim, num_frames):
        self.manifest = manifest_df
        self.resize_dim = resize_dim
        self.num_frames = num_frames

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_path = row["fragment_path"]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.resize_dim[1], self.resize_dim[0]))
                frames.append(frame)
            else:
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(
                        np.zeros(
                            (self.resize_dim[0], self.resize_dim[1], 3), dtype=np.uint8
                        )
                    )

        cap.release()

        video_tensor = torch.from_numpy(np.array(frames, dtype=np.float32)) / 255.0
        video_tensor = video_tensor.permute(3, 0, 1, 2)

        sbp = torch.tensor(row["sbp"], dtype=torch.float32)
        dbp = torch.tensor(row["dbp"], dtype=torch.float32)
        labels = torch.stack([sbp, dbp])

        return video_tensor, labels


# --- 3. Model Architecture ---
class VideoBPNet(nn.Module):
    def __init__(self, num_clips, clip_len):
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

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size * self.num_clips, 3, self.clip_len, *RESIZE_DIM)

        features = self.cnn(x)
        features = features.view(batch_size, self.num_clips, -1)

        lstm_out, _ = self.lstm(features)
        last_step_out = lstm_out[:, -1, :]

        out = self.fc(last_step_out)
        return out


# --- 4. Plotting and History Functions ---
def plot_and_save_history(history, fold, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(f"Fold {fold} Training History", fontsize=16)

    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss (MSE)")
    ax1.plot(history["epoch"], history["test_loss"], label="Test Loss (MSE)")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Mean Squared Error Loss vs. Epochs")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["epoch"], history["train_mae"], label="Train MAE")
    ax2.plot(history["epoch"], history["test_mae"], label="Test MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error (MAE)")
    ax2.set_title("Mean Absolute Error vs. Epochs")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_subject_metrics(subject_metrics_df, fold, save_path):
    subject_metrics_df = subject_metrics_df.sort_values(by="subject_id")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    fig.suptitle(f"Fold {fold} - Per-Subject Metrics", fontsize=16)

    subjects = subject_metrics_df["subject_id"]

    ax1.bar(subjects, subject_metrics_df["mse"])
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Mean Squared Error per Subject")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y")

    ax2.bar(subjects, subject_metrics_df["mae"])
    ax2.set_ylabel("MAE")
    ax2.set_title("Mean Absolute Error per Subject")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


# --- 5. Training and Evaluation Functions ---
def train_and_test(
    model,
    train_loader,
    test_loader,
    optimizer,
    mse_criterion,
    mae_criterion,
    epochs,
    writer,
):
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_mae": [],
        "test_mae": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss, train_mae = 0.0, 0.0

        # Wrap train_loader with tqdm for a progress bar
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False
        )
        for videos, labels in train_pbar:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = mse_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * videos.size(0)

            with torch.no_grad():
                mae = mae_criterion(outputs, labels)
                train_mae += mae.item() * videos.size(0)

            # Update the progress bar description with live loss
            train_pbar.set_postfix({"MSE": loss.item(), "MAE": mae.item()})

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        model.eval()
        test_loss, test_mae = 0.0, 0.0
        with torch.no_grad():
            # Wrap test_loader with tqdm for a progress bar
            test_pbar = tqdm(
                test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Test]", leave=False
            )
            for videos, labels in test_pbar:
                videos, labels = videos.to(DEVICE), labels.to(DEVICE)
                outputs = model(videos)
                loss = mse_criterion(outputs, labels)
                mae = mae_criterion(outputs, labels)
                test_loss += loss.item() * videos.size(0)
                test_mae += mae.item() * videos.size(0)
                test_pbar.set_postfix({"MSE": loss.item(), "MAE": mae.item()})

        test_loss /= len(test_loader.dataset)
        test_mae /= len(test_loader.dataset)

        print(
            f"Epoch {epoch + 1}/{epochs} Summary | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}"
        )

        # Log to TensorBoard
        writer.add_scalar("Loss/Train_MSE", train_loss, epoch + 1)
        writer.add_scalar("Loss/Test_MSE", test_loss, epoch + 1)
        writer.add_scalar("MAE/Train", train_mae, epoch + 1)
        writer.add_scalar("MAE/Test", test_mae, epoch + 1)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_mae"].append(train_mae)
        history["test_mae"].append(test_mae)
    return history


def evaluate_per_subject(
    model, full_dataset, test_subject_map, mse_criterion, mae_criterion
):
    model.eval()
    subject_metrics = []

    with torch.no_grad():
        # Wrap the subject loop with tqdm for progress
        subject_pbar = tqdm(
            test_subject_map.items(), desc="Evaluating Subjects", leave=False
        )
        for subject_id, indices in subject_pbar:
            subject_pbar.set_postfix({"Subject": subject_id})
            subject_subset = Subset(full_dataset, indices)
            subject_loader = DataLoader(
                subject_subset, batch_size=BATCH_SIZE, shuffle=False
            )

            subject_mse, subject_mae = 0.0, 0.0
            for videos, labels in subject_loader:
                videos, labels = videos.to(DEVICE), labels.to(DEVICE)
                outputs = model(videos)
                subject_mse += mse_criterion(outputs, labels).item() * videos.size(0)
                subject_mae += mae_criterion(outputs, labels).item() * videos.size(0)

            avg_mse = subject_mse / len(subject_subset)
            avg_mae = subject_mae / len(subject_subset)

            subject_metrics.append(
                {"subject_id": subject_id, "mse": avg_mse, "mae": avg_mae}
            )

    return pd.DataFrame(subject_metrics)


# --- 6. Main Execution with K-Fold Cross-Validation ---
if __name__ == "__main__":
    print("\n--- Model Architecture ---")
    temp_model = VideoBPNet(num_clips=NUM_CLIPS, clip_len=CLIP_LEN)
    print(temp_model)
    with open(os.path.join(RESULTS_DIR, "model_architecture.txt"), "w") as f:
        f.write(str(temp_model))
    print("Model architecture saved to results/model_architecture.txt")
    del temp_model

    manifest_df = pd.read_csv(MANIFEST_FILE)
    subjects = manifest_df["subject_id"].unique()
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    full_dataset = VideoBPDataset(
        manifest_df=manifest_df, num_frames=NUM_FRAMES, resize_dim=RESIZE_DIM
    )

    fold_results = []

    for fold, (train_subject_indices, test_subject_indices) in enumerate(
        kf.split(subjects)
    ):
        print(f"\n{'=' * 25} FOLD {fold + 1}/{NUM_FOLDS} {'=' * 25}")

        train_subjects = subjects[train_subject_indices]
        test_subjects = subjects[test_subject_indices]

        train_indices = manifest_df[
            manifest_df["subject_id"].isin(train_subjects)
        ].index.tolist()
        test_indices = manifest_df[
            manifest_df["subject_id"].isin(test_subjects)
        ].index.tolist()

        test_subject_map = {
            subj: manifest_df[manifest_df["subject_id"] == subj].index.tolist()
            for subj in test_subjects
        }

        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)

        print(
            f"Training on {len(train_subjects)} subjects ({len(train_subset)} videos)."
        )
        print(f"Testing on {len(test_subjects)} subjects ({len(test_subset)} videos).")

        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        print("Initializing a new model and TensorBoard writer for this fold...")
        model = VideoBPNet(num_clips=NUM_CLIPS, clip_len=CLIP_LEN).to(DEVICE)
        writer = SummaryWriter(
            log_dir=os.path.join(RESULTS_DIR, "runs", f"fold_{fold + 1}")
        )

        # Add model graph to TensorBoard
        sample_videos, _ = next(iter(train_loader))
        writer.add_graph(model, sample_videos.to(DEVICE))

        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        history = train_and_test(
            model,
            train_loader,
            test_loader,
            optimizer,
            mse_criterion,
            mae_criterion,
            epochs=NUM_EPOCHS,
            writer=writer,
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(
            os.path.join(RESULTS_DIR, f"fold_{fold + 1}_history.csv"), index=False
        )
        plot_and_save_history(
            history, fold + 1, os.path.join(RESULTS_DIR, f"fold_{fold + 1}_metrics.png")
        )
        print(f"Saved epoch history and plot for fold {fold + 1}.")

        subject_metrics_df = evaluate_per_subject(
            model, full_dataset, test_subject_map, mse_criterion, mae_criterion
        )
        subject_metrics_df.to_csv(
            os.path.join(RESULTS_DIR, f"fold_{fold + 1}_subject_metrics.csv"),
            index=False,
        )
        plot_subject_metrics(
            subject_metrics_df,
            fold + 1,
            os.path.join(RESULTS_DIR, f"fold_{fold + 1}_subject_metrics.png"),
        )
        print(f"Saved per-subject metrics and plot for fold {fold + 1}.")

        fold_results.append(
            {
                "fold": fold + 1,
                "test_loss_mse": history["test_loss"][-1],
                "test_mae": history["test_mae"][-1],
            }
        )

        writer.close()

    print(f"\n{'=' * 20} CROSS-VALIDATION SUMMARY {'=' * 20}")
    results_df = pd.DataFrame(fold_results)

    avg_mse = results_df["test_loss_mse"].mean()
    std_mse = results_df["test_loss_mse"].std()
    avg_mae = results_df["test_mae"].mean()
    std_mae = results_df["test_mae"].std()

    print(
        f"Average Test MSE across {NUM_FOLDS} folds: {avg_mse:.4f} (std: {std_mse:.4f})"
    )
    print(
        f"Average Test MAE across {NUM_FOLDS} folds: {avg_mae:.4f} (std: {std_mae:.4f})"
    )

    print("\nIndividual fold results:")
    print(results_df.to_string(index=False))
