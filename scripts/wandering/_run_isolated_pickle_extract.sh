#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 SOURCE_PKL OUTPUT_DIR EXTRACTOR_SCRIPT PYTHON_ENV EXPECTED_SHA256" >&2
  exit 64
fi

source_file="$1"
output_dir="$2"
extractor_script="$3"
python_env="$4"
expected_sha256="$5"

for path in "$source_file" "$output_dir" "$extractor_script" "$python_env"; do
  if [[ "$path" != /* ]]; then
    echo "all isolation paths must be absolute: $path" >&2
    exit 64
  fi
done
if [[ ! -f "$source_file" || ! -f "$extractor_script" ]]; then
  echo "source pickle and extractor script must be regular files" >&2
  exit 66
fi
if [[ ! -d "$output_dir" || ! -x "$python_env/bin/python" ]]; then
  echo "output directory or isolated Python environment is unavailable" >&2
  exit 66
fi
if [[ ! "$expected_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "expected SHA-256 must be lowercase hexadecimal" >&2
  exit 64
fi

cd /
mount --make-rprivate /
mount -t tmpfs -o size=64m,nosuid,nodev,noexec tmpfs /opt
isolation_root="/opt/wandering_pickle_isolation"
mkdir -p \
  "$isolation_root/input" \
  "$isolation_root/output" \
  "$isolation_root/runtime" \
  "$isolation_root/script"
touch "$isolation_root/input/patterns_dataset.pkl"
touch "$isolation_root/script/extractor.py"

mount --bind "$source_file" "$isolation_root/input/patterns_dataset.pkl"
mount -o remount,bind,ro,nosuid,nodev,noexec "$isolation_root/input/patterns_dataset.pkl"
mount --bind "$extractor_script" "$isolation_root/script/extractor.py"
mount -o remount,bind,ro,nosuid,nodev,noexec "$isolation_root/script/extractor.py"
mount --bind "$python_env" "$isolation_root/runtime"
mount -o remount,bind,ro,nosuid,nodev "$isolation_root/runtime"
mount --bind "$output_dir" "$isolation_root/output"
mount -o remount,bind,rw,nosuid,nodev,noexec "$isolation_root/output"

# Hide host user data, Windows drives, session sockets, and host temporary files.
mount -t tmpfs -o size=1m,nosuid,nodev,noexec tmpfs /home
mount -t tmpfs -o size=1m,nosuid,nodev,noexec tmpfs /mnt
if [[ -d /run/user ]]; then
  mount -t tmpfs -o size=1m,nosuid,nodev,noexec tmpfs /run/user
fi
mount -t tmpfs -o size=64m,nosuid,nodev,noexec tmpfs /tmp

umask 077
ulimit -t 90
ulimit -v 2097152
ulimit -f 65536
ulimit -n 64

exec env -i \
  HOME=/nonexistent \
  PATH=/usr/bin:/bin \
  WANDERING_PICKLE_ISOLATED=1 \
  "$isolation_root/runtime/bin/python" -I -B \
  "$isolation_root/script/extractor.py" \
  --input "$isolation_root/input/patterns_dataset.pkl" \
  --output-jsonl "$isolation_root/output/rows.jsonl" \
  --output-metadata "$isolation_root/output/metadata.json" \
  --expected-sha256 "$expected_sha256"
