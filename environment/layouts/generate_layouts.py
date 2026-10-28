import argparse
from pathlib import Path

import numpy as np

SEED = 42

NUM_TRAIN_LAYOUTS = 30
NUM_TEST_LAYOUTS = 10
NUM_VAL_LAYOUTS = 10

CORRIDOR_LENGTH = 30.0

# Every layout draws its own corridor width from this range.
MIN_CORRIDOR_WIDTH = 4.0
MAX_CORRIDOR_WIDTH = 8.0

# Every layout draws its own pillar count from this range.
MIN_PILLARS = 3
MAX_PILLARS = 12

# Every pillar draws its own radius from this range.
MIN_RADIUS = 0.3
MAX_RADIUS = 0.7

START_CLEARANCE = 3.0
GOAL_CLEARANCE = 3.0

WALL_CLEARANCE = 0.25
PILLAR_CLEARANCE = 0.75

DEFAULT_LAYOUTS_DIR = Path(__file__).resolve().parents[2] / "layouts"


def generate_layout(
    rng,
    corridor_length,
    min_corridor_width=MIN_CORRIDOR_WIDTH,
    max_corridor_width=MAX_CORRIDOR_WIDTH,
    min_pillars=MIN_PILLARS,
    max_pillars=MAX_PILLARS,
    min_radius=MIN_RADIUS,
    max_radius=MAX_RADIUS,
):
    """Generate one corridor layout with a random width, pillar count, and radii.

    The corridor width is drawn uniformly from
    ``[min_corridor_width, max_corridor_width]``, the number of pillars
    uniformly from ``[min_pillars, max_pillars]`` (inclusive), and each
    pillar's radius uniformly from ``[min_radius, max_radius]``, so a layout
    set mixes narrow and wide corridors with sparse and dense obstacle
    fields of varying size. The returned dict stores the ``width`` together
    with the ``(num_pillars, 3)`` array of ``(x, y, radius)`` pillars;
    ``CorridorEnvironment`` adopts both when it loads the layout, so start
    and goal sampling use the geometry the episode actually runs on.
    """
    if min_pillars < 0 or max_pillars < min_pillars:
        raise ValueError("require 0 <= min_pillars <= max_pillars")
    if min_corridor_width <= 0 or max_corridor_width < min_corridor_width:
        raise ValueError(
            "require 0 < min_corridor_width <= max_corridor_width"
        )
    if min_radius <= 0 or max_radius < min_radius:
        raise ValueError("require 0 < min_radius <= max_radius")
    # The widest pillar plus its wall clearance must fit across the corridor.
    if min_corridor_width < 2.0 * (max_radius + WALL_CLEARANCE):
        raise ValueError(
            f"min_corridor_width must be at least "
            f"{2.0 * (max_radius + WALL_CLEARANCE):.2f} m"
        )

    corridor_width = float(
        rng.uniform(min_corridor_width, max_corridor_width)
    )
    num_pillars = int(rng.integers(min_pillars, max_pillars + 1))

    pillars = []

    max_attempts = 1000
    attempts = 0

    while(len(pillars) < num_pillars and attempts < max_attempts):
        attempts += 1

        radius = rng.uniform(min_radius, max_radius)

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

    return {
        "width": corridor_width,
        "pillars": np.asarray(pillars, dtype=np.float32),
    }

def generate_dataset(
    seed,
    num_layouts,
    min_corridor_width=MIN_CORRIDOR_WIDTH,
    max_corridor_width=MAX_CORRIDOR_WIDTH,
    min_pillars=MIN_PILLARS,
    max_pillars=MAX_PILLARS,
    min_radius=MIN_RADIUS,
    max_radius=MAX_RADIUS,
):
    rng = np.random.default_rng(seed)

    layouts = []

    for _ in range(num_layouts):
        layout = generate_layout(
            rng=rng,
            corridor_length=CORRIDOR_LENGTH,
            min_corridor_width=min_corridor_width,
            max_corridor_width=max_corridor_width,
            min_pillars=min_pillars,
            max_pillars=max_pillars,
            min_radius=min_radius,
            max_radius=max_radius,
        )
        layouts.append(layout)

    # Layouts have different pillar counts, so the set is a ragged object
    # array of per-layout dicts rather than a fixed-shape array.
    return np.asarray(layouts, dtype=object)


def generate_layouts(
    name=None,
    save_dir=DEFAULT_LAYOUTS_DIR,
    num_train_layouts=NUM_TRAIN_LAYOUTS,
    num_test_layouts=NUM_TEST_LAYOUTS,
    num_val_layouts=NUM_VAL_LAYOUTS,
    min_pillars=MIN_PILLARS,
    max_pillars=MAX_PILLARS,
    min_corridor_width=MIN_CORRIDOR_WIDTH,
    max_corridor_width=MAX_CORRIDOR_WIDTH,
    min_radius=MIN_RADIUS,
    max_radius=MAX_RADIUS,
    seed=SEED,
):
    """Generate and save train, test, and validation corridor layouts.

    Each layout samples its own pillar count, corridor width, and pillar
    radii, so the saved set covers a range of geometries. Layouts are saved
    as ``<name>_{train,test,validation}.npy`` in ``save_dir``; each entry is
    a dict with the layout's ``width`` and ``(x, y, radius)`` ``pillars``.
    """
    if min(num_train_layouts, num_test_layouts, num_val_layouts) < 0:
        raise ValueError("layout counts must be non-negative")

    name = name or "corridor_layouts"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_args = {
        "min_corridor_width": min_corridor_width,
        "max_corridor_width": max_corridor_width,
        "min_pillars": min_pillars,
        "max_pillars": max_pillars,
        "min_radius": min_radius,
        "max_radius": max_radius,
    }
    train_layouts = generate_dataset(
        seed=seed,
        num_layouts=num_train_layouts,
        **dataset_args,
    )
    test_layouts = generate_dataset(
        seed=seed + 1,
        num_layouts=num_test_layouts,
        **dataset_args,
    )
    validation_layouts = generate_dataset(
        seed=seed + 2,
        num_layouts=num_val_layouts,
        **dataset_args,
    )

    paths = {
        "train": save_dir / f"{name}_train.npy",
        "test": save_dir / f"{name}_test.npy",
        "validation": save_dir / f"{name}_validation.npy",
    }
    np.save(paths["train"], train_layouts)
    np.save(paths["test"], test_layouts)
    np.save(paths["validation"], validation_layouts)

    for label, dataset in (
        ("train", train_layouts),
        ("test", test_layouts),
        ("validation", validation_layouts),
    ):
        pillar_counts = [len(layout["pillars"]) for layout in dataset]
        widths = [layout["width"] for layout in dataset]
        radii = [
            pillar[2] for layout in dataset for pillar in layout["pillars"]
        ]
        print(
            f"Saved {len(dataset)} {label} layouts | "
            f"pillars {min(pillar_counts, default=0)}-"
            f"{max(pillar_counts, default=0)} | "
            f"width {min(widths, default=0.0):.2f}-"
            f"{max(widths, default=0.0):.2f} m | "
            f"radius {min(radii, default=0.0):.2f}-"
            f"{max(radii, default=0.0):.2f} m"
        )

    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Generate corridor layouts.")
    parser.add_argument(
        "--name",
        default=None,
        help="Filename prefix (default: corridor_layouts).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_LAYOUTS_DIR,
        help=f"Directory for generated layouts (default: {DEFAULT_LAYOUTS_DIR}).",
    )
    parser.add_argument("--num-train-layouts", type=int, default=NUM_TRAIN_LAYOUTS)
    parser.add_argument("--num-test-layouts", type=int, default=NUM_TEST_LAYOUTS)
    parser.add_argument("--num-val-layouts", type=int, default=NUM_VAL_LAYOUTS)
    parser.add_argument("--min-pillars", type=int, default=MIN_PILLARS)
    parser.add_argument("--max-pillars", type=int, default=MAX_PILLARS)
    parser.add_argument(
        "--min-width", type=float, default=MIN_CORRIDOR_WIDTH
    )
    parser.add_argument(
        "--max-width", type=float, default=MAX_CORRIDOR_WIDTH
    )
    parser.add_argument("--min-radius", type=float, default=MIN_RADIUS)
    parser.add_argument("--max-radius", type=float, default=MAX_RADIUS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_layouts(
        name=args.name,
        save_dir=args.save_dir,
        num_train_layouts=args.num_train_layouts,
        num_test_layouts=args.num_test_layouts,
        num_val_layouts=args.num_val_layouts,
        min_pillars=args.min_pillars,
        max_pillars=args.max_pillars,
        min_corridor_width=args.min_width,
        max_corridor_width=args.max_width,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )

