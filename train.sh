#!/usr/bin/env bash
# Start training for the RCRL TD3 or plain TD3 agent.
#
# Usage:
#   ./train.sh <rcrl|rcrl-vec|td3> [trainer flags...]
#
# The agent selects the trainer (train_scripts/train_rcrl_td3.py,
# train_scripts/train_rcrl_td3_vec.py, or train_scripts/train_td3.py); all
# following flags are passed through to it unchanged, so
# `./train.sh rcrl --help` lists the full flag set.
#
# Examples:
#   ./train.sh rcrl --name rcrl_lidar
#   ./train.sh rcrl-vec --name rcrl_vec --num-envs 32
#   ./train.sh td3 --name td3_run
#   ./train.sh rcrl --name quick --max-timesteps 8 --eval-freq 0
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<EOF
Usage: $(basename "$0") <rcrl|rcrl-vec|td3> [trainer flags...]

  rcrl       Train the reachability-constrained TD3 agent
             (python -m train_scripts.train_rcrl_td3)
  rcrl-vec   Vectorized RCRL-TD3 trainer: parallel environments and a
             device-resident replay buffer, same algorithm and
             update-to-data ratio as 'rcrl'
             (python -m train_scripts.train_rcrl_td3_vec)
  td3        Train the plain TD3 baseline
             (python -m train_scripts.train_td3)

Trainer flags are passed through unchanged; see '<agent> --help'.
EOF
}

if [ $# -lt 1 ]; then
    usage >&2
    exit 1
fi

agent="$1"
case "$agent" in
    -h|--help)
        usage
        exit 0
        ;;
    rcrl|rcrl_td3)
        module="train_scripts.train_rcrl_td3"
        ;;
    rcrl-vec|rcrl_vec)
        module="train_scripts.train_rcrl_td3_vec"
        ;;
    td3)
        module="train_scripts.train_td3"
        ;;
    *)
        echo "Unknown agent '$agent' (expected 'rcrl' or 'td3')" >&2
        usage >&2
        exit 1
        ;;
esac

shift
exec python -m "$module" "$@"
