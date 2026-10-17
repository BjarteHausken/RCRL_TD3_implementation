import argparse
from pathlib import Path

import numpy as np

SEED = 42

NUM_TRAIN_LAYOUTS = 30
NUM_TEST_LAYOUTS = 10
NUM_VAL_LAYOUTS = 10

CORRIDOR_LENGTH = 30.0
CORRIDOR_WIDTH = 5.0

NUM_PILLARS = 7

MIN_RADIUS = 0.5
MAX_RADIUS = 0.5

START_CLEARANCE = 3.0
GOAL_CLEARANCE = 3.0

WALL_CLEARANCE = 0.25
PILLAR_CLEARANCE = 0.75

DEFAULT_LAYOUTS_DIR = Path(__file__).resolve().parents[2] / "layouts"


def generate_layout(
    rng,
    corridor_length,
    corridor_width,
    num_pillars,
):
    pillars = []

    max_attempts = 1000
    attempts = 0

    while(len(pillars) < num_pillars and attempts < max_attempts):
        attempts += 1
        
        radius = rng.uniform(MIN_RADIUS, MAX_RADIUS)

        x = rng.uniform(START_CLEARANCE, corridor_length - GOAL_CLEARANCE)
        
        max_y = corridor_width / 2 - radius - WALL_CLEARANCE

        y = rng.uniform(-max_y, max_y)

        valid = True

        for px, py, pr in pillars:
            distance = np.sqrt((x - px) ** 2 + (y - py) ** 2)

            required_distance = radius + pr + PILLAR_CLEARANCE

            if distance < required_distance:
                valid = False
                break

        if valid:
            pillars.append([x, y, radius])
            
    if len(pillars) < num_pillars:
        raise RuntimeError("Failed to generate valid layout after maximum attempts.")

    return np.asarray(pillars, dtype=np.float32)

def generate_dataset(
    seed,
    num_layouts,
    num_pillars,
):
    rng = np.random.default_rng(seed)

    layouts = []

    for _ in range(num_layouts):
        layout = generate_layout(
            rng=rng,
            corridor_length=CORRIDOR_LENGTH,
            corridor_width=CORRIDOR_WIDTH,
            num_pillars=num_pillars,
        )
        layouts.append(layout)

    return np.asarray(layouts, dtype=np.float32)


def generate_layouts(
    name=None,
    num_pillars=NUM_PILLARS,
    save_dir=DEFAULT_LAYOUTS_DIR,
    num_train_layouts=NUM_TRAIN_LAYOUTS,
    num_test_layouts=NUM_TEST_LAYOUTS,
    num_val_layouts=NUM_VAL_LAYOUTS,
    seed=SEED,
):
    """Generate and save train, test, and validation corridor layouts.

    Layouts are saved as ``<name>_{train,test,validation}.npy`` in
    ``save_dir``. When no name is supplied, it is derived from the number of
    pillars (for example, ``corridor_layouts_7``).
    """
    if num_pillars < 0:
        raise ValueError("num_pillars must be non-negative")
    if min(num_train_layouts, num_test_layouts, num_val_layouts) < 0:
        raise ValueError("layout counts must be non-negative")

    name = name or f"corridor_layouts_{num_pillars}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_layouts = generate_dataset(
        seed=seed,
        num_layouts=num_train_layouts,
        num_pillars=num_pillars,
    )
    test_layouts = generate_dataset(
        seed=seed + 1,
        num_layouts=num_test_layouts,
        num_pillars=num_pillars,
    )
    validation_layouts = generate_dataset(
        seed=seed + 2,
        num_layouts=num_val_layouts,
        num_pillars=num_pillars,
    )

    paths = {
        "train": save_dir / f"{name}_train.npy",
        "test": save_dir / f"{name}_test.npy",
        "validation": save_dir / f"{name}_validation.npy",
    }
    np.save(paths["train"], train_layouts)
    np.save(paths["test"], test_layouts)
    np.save(paths["validation"], validation_layouts)

    print(f"Saved {len(train_layouts)} train layouts")
    print(f"Saved {len(test_layouts)} test layouts")
    print(f"Saved {len(validation_layouts)} validation layouts")
    print("Train layouts shape:", train_layouts.shape)
    print("Test layouts shape:", test_layouts.shape)
    print("Validation layouts shape:", validation_layouts.shape)

    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Generate corridor layouts.")
    parser.add_argument(
        "--name",
        help="Filename prefix (default: corridor_layouts_<num-pillars>).",
    )
    parser.add_argument("--num-pillars", type=int, default=NUM_PILLARS)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_LAYOUTS_DIR,
        help=f"Directory for generated layouts (default: {DEFAULT_LAYOUTS_DIR}).",
    )
    parser.add_argument("--num-train-layouts", type=int, default=NUM_TRAIN_LAYOUTS)
    parser.add_argument("--num-test-layouts", type=int, default=NUM_TEST_LAYOUTS)
    parser.add_argument("--num-val-layouts", type=int, default=NUM_VAL_LAYOUTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_layouts(
        name=args.name,
        num_pillars=args.num_pillars,
        save_dir=args.save_dir,
        num_train_layouts=args.num_train_layouts,
        num_test_layouts=args.num_test_layouts,
        num_val_layouts=args.num_val_layouts,
    )

