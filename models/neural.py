from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .logistic import (
        CLEANED_TRAIN_DIR,
        CANDIDATE_POOL_SIZE,
        CANDIDATES_PER_OBSERVATION,
        FEATURE_NAMES,
        STUDENT_TEST_DIR,
        TRAIN_DIR,
        build_trip_candidates,
    )
else:
    from logistic import (
        CLEANED_TRAIN_DIR,
        CANDIDATE_POOL_SIZE,
        CANDIDATES_PER_OBSERVATION,
        FEATURE_NAMES,
        STUDENT_TEST_DIR,
        TRAIN_DIR,
        build_trip_candidates,
    )


MODEL_PATH = Path(__file__).resolve().parent / "neural_map_matcher.pt"
RESULTS_PATH = Path(__file__).resolve().parent / "neural_student_test_results.csv"
HISTORY_PATH = Path(__file__).resolve().parent / "neural_training_history.csv"

SEED = 42
HIDDEN_DIM = 64
DROPOUT = 0.2
BATCH_SIZE = 2048
EPOCHS = 20
LEARNING_RATE = 0.001
INFERENCE_BATCH_SIZE = 8192


class ShallowMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.layers(features)


def train_model(x_train, y_train, device):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train_scaled),
        torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1)),
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = ShallowMLP(len(FEATURE_NAMES), HIDDEN_DIM, DROPOUT).to(device)
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    positive_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)

        epoch_loss = total_loss / len(train_dataset)
        history.append({"epoch": epoch + 1, "training_loss": epoch_loss})
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss:.4f}")

    return model, scaler, history


def score_candidates(model, scaler, features, device):
    flat_features = features.reshape(-1, len(FEATURE_NAMES))
    scaled_features = scaler.transform(flat_features).astype(np.float32)
    scores = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(scaled_features), INFERENCE_BATCH_SIZE):
            batch = torch.from_numpy(
                scaled_features[start : start + INFERENCE_BATCH_SIZE]
            ).to(device)
            scores.append(model(batch).squeeze(1).cpu().numpy())

    return np.concatenate(scores).reshape(features.shape[:2])


def main():
    if not CLEANED_TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Cleaned training data not found: {CLEANED_TRAIN_DIR}. "
            "Run Student folder\\cleaning.py first."
        )
    if not TRAIN_DIR.is_dir() or not STUDENT_TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected both training and student-test folders under "
            f"{TRAIN_DIR.parent}"
        )

    train_trip_dirs = sorted(path for path in TRAIN_DIR.iterdir() if path.is_dir())
    test_trip_dirs = sorted(path for path in STUDENT_TEST_DIR.iterdir() if path.is_dir())
    if not train_trip_dirs or not test_trip_dirs:
        raise FileNotFoundError("Training or student-test trip folders were not found.")

    started = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    training_features = []
    training_labels = []
    train_coverage = []
    for trip_dir in train_trip_dirs:
        features, _, _, labels, covered = build_trip_candidates(
            trip_dir,
            use_cleaned_track=True,
        )
        train_coverage.append(float(covered.mean()))
        if covered.any():
            training_features.append(
                features[covered].reshape(-1, len(FEATURE_NAMES))
            )
            training_labels.append(labels[covered].reshape(-1).astype(np.int8))

    if not training_features:
        raise ValueError("No training GPS observations had a reference-route candidate.")

    x_train = np.concatenate(training_features)
    y_train = np.concatenate(training_labels)
    if np.unique(y_train).size != 2:
        raise ValueError("Candidate training data must contain both positive and negative labels.")

    model, scaler, history = train_model(x_train, y_train, device)

    trip_results = []
    for trip_dir in test_trip_dirs:
        features, candidate_ids, target_ids, _, covered = build_trip_candidates(
            trip_dir,
            use_cleaned_track=False,
        )
        candidate_scores = score_candidates(model, scaler, features, device)
        selected = np.argmax(candidate_scores, axis=1)
        predicted_ids = candidate_ids[np.arange(len(candidate_ids)), selected]
        correct = predicted_ids == target_ids
        covered_count = int(covered.sum())
        correct_covered = int((correct & covered).sum())
        trip_results.append(
            {
                "trip_id": trip_dir.name,
                "observations": len(target_ids),
                "candidate_covered_observations": covered_count,
                "correct_on_covered_observations": correct_covered,
                "candidate_coverage": float(covered.mean()),
                "top1_accuracy_on_covered": (
                    correct_covered / covered_count if covered_count else float("nan")
                ),
                "top1_accuracy_all_observations": float(correct.mean()),
            }
        )

    result_frame = pd.DataFrame(trip_results)
    total_observations = int(result_frame["observations"].sum())
    total_covered = int(result_frame["candidate_covered_observations"].sum())
    total_correct_covered = int(
        result_frame["correct_on_covered_observations"].sum()
    )
    result_frame.to_csv(RESULTS_PATH, index=False)
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    torch.save(
        {
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "input_dim": len(FEATURE_NAMES),
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "feature_names": FEATURE_NAMES,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "candidates_per_observation": CANDIDATES_PER_OBSERVATION,
            "label_method": "nearest reference-route arc to each GPS observation",
            "training_trip_ids": [path.name for path in train_trip_dirs],
            "seed": SEED,
            "epochs": EPOCHS,
        },
        MODEL_PATH,
    )

    print(f"Training trips: {len(train_trip_dirs)}")
    print(f"Training candidate rows: {len(y_train):,}")
    print(f"Training candidate positive rate: {y_train.mean():.4f}")
    print(f"Mean training candidate coverage: {np.mean(train_coverage):.2%}")
    print(f"Student-test trips: {len(test_trip_dirs)}")
    print(result_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "Overall student-test candidate coverage: "
        f"{total_covered / total_observations:.2%}"
    )
    print(
        "Overall top-1 accuracy on covered observations: "
        f"{total_correct_covered / total_covered:.4f}"
        if total_covered
        else "Overall top-1 accuracy on covered observations: N/A"
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Student-test results saved to: {RESULTS_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")
    print(f"Elapsed time: {time.perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    main()
