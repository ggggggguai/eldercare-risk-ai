# Third-Party Notices: Micro-Expression Module

This notice applies to the pinned, audit-only reference snapshots stored under
`third_party/microexpression/`. These snapshots are evidence for source review
and reimplementation. They are not imported or executed by the application.

## CausalNet

- Project: CausalNet
- Repository: https://github.com/tony19980810/CausalNet
- Pinned commit: `7bdf163face030eeae4358fc7638cf538305acce`
- License: MIT
- Copyright: Copyright (c) 2025 tony19980810
- License copy: `third_party/microexpression/CausalNet/LICENSE`
- Reference snapshot: `third_party/microexpression/CausalNet/reference/`

The project may adapt architectural ideas from this source. The upstream
training script is not used because it evaluates the held-out test subject on
every epoch and uses that result for checkpoint selection and early stopping.
The public 28x28 model path also contains a 64/256-dimensional tensor mismatch.
Any project implementation must record its correction, preserve this notice,
set `upstream_bit_exact=false`, and use the project's strict subject-level
Nested-LOSO evaluation.

## MEGC2024-CODE

- Project: MEGC2024-CODE
- Repository: https://github.com/hitheyuhong/MEGC2024-CODE
- Pinned commit: `826aa718efcc44920a1f6ae373350fade7ed92da`
- License: Apache-2.0
- License copy: `third_party/microexpression/MEGC2024-CODE/LICENSE`
- Reference snapshot: `third_party/microexpression/MEGC2024-CODE/reference/`

The project may adapt concepts for spotting windows, temporal filtering,
interval expansion, interval suppression, and evaluation scaffolding. The
upstream fixed five-fold and eight-epoch protocol does not replace the
project's subject-isolated evaluation. Dataset-specific thresholds and paths
must not be copied into deployment code without training-side validation.

## Usage Boundary

The audit-only reference snapshots remain under their respective upstream
licenses. New project code must be independently reviewed, attributed where it
contains adapted portions, and kept under the algorithm package. No source,
feature engineering, model inference, spotting threshold, or scoring logic may
be moved into `backend/`.
