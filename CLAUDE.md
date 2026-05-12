# CLAUDE.md — Project-Specific Coding & Execution Guidelines

This document contains behavioral guidelines, coordinate conventions, validation standards, and module roles tailored specifically for this 3D DFN Modeling project. 

Always adhere to these rules. Clarity is more valuable than cleverness. Geometry and DFN logic should be easy to inspect and debug.

---

## 1. Think Before Coding
- **Do not assume** the user's intended algorithm, coordinate convention, or validation metric.
- **State the working assumptions briefly** before implementing.
  - *Examples:*
    - Whether the tunnel axis is treated as the $x$ direction (standard: yes).
    - Whether excavation faces are planes of constant $x$.
    - Whether trace endpoints are in local face coordinates $[y, z]$ or global 3D coordinates $[x, y, z]$.
    - Whether fracture geometry is represented as an infinite plane, bounded disc, polygon, or clipped trace.
    - Whether CCA should use 6-connectivity or 26-connectivity.
    - Whether the target metric is $P_{21}$, $P_{32}$, trace count, trace length distribution, orientation distribution, or block count.
- **If multiple interpretations are possible, do not silently choose one.** Present them and seek/state assumptions.
  - *Examples:*
    - "Two-face trace matching" can mean matching 2D traces between Face $n-1$ and Face $n$ and fitting a 3D plane, using already provided 3D trace coordinates to directly infer planes, or generating stochastic fractures constrained by observed traces.
    - "Improve $P_{21}$ error" can mean reducing deterministic fracture over-generation, recalibrating stochastic $P_{30}$, adjusting trace clipping, or changing the objective function.
- **If a simpler approach is sufficient, say so** and implement the simpler approach. Do not build a large framework unless the request explicitly requires it.

## 2. Simplicity First
- **Write the minimum code** required to solve the requested problem.
- **Avoid:**
  - New abstractions for one-off logic.
  - New configuration systems unless requested.
  - New classes when a small function is enough.
  - Broad refactors or large helper modules.
  - Extra visualizations not requested.
  - Speculative error handling for impossible states.
- **Double-check before finalizing:** *Could this change be 50 lines instead of 200?* If yes, simplify.

## 3. Surgical Changes Only
- **Touch only the files required** for the requested task.
- **Do not "improve"** adjacent code, formatting, comments, names, or structure unless directly necessary.
- **Match the existing style** even if a different style would be preferred.
- **If unrelated dead code or suspicious logic is noticed**, mention it in the response instead of deleting it.
- **Every changed line must trace directly** to the user's request.
- **When a change creates unused imports, variables, or helpers**, remove *only* the unused code introduced by that change. Do not remove pre-existing unused code unless explicitly asked.

## 4. Repository Architecture Awareness
Use the existing module boundaries. Do not read or rewrite the entire codebase unless necessary.

### Expected Module Roles:
- **`detect_blocks_gpu.py`** (or main entry point script): Main pipeline and CLI entry point.
  - *Responsibilities:* Load HDF5 DFN/tunnel input; call tunnel voxelization; call voxel classification and CCA; save block labels and metadata; trigger visualization outputs.
  - *Modify only when* changing pipeline flow, CLI arguments, input/output paths, or orchestration logic.
- **`block_detector.py`**: Core voxel classification and CCA logic.
  - *Responsibilities:* Classify voxels as rock, fracture, or tunnel using fracture disc/plane equations and bounding boxes; run connected component analysis (CCA); filter blocks that touch the tunnel but do not touch the outer model boundary; compute block statistics.
  - *Modify when* changing CCA connectivity, fracture voxelization logic, block filtering criteria, GPU/CPU voxel operations, or block statistics. (Be especially careful: small geometric changes strongly affect block count and $P_{21}$/$P_{32}$ behavior).
- **`tunnel_geometry.py`**: Tunnel voxelization.
  - *Responsibilities:* Convert 2D tunnel polygon geometry in the Y-Z plane into a 3D tunnel mask extruded along the $X$ direction; determine which voxels are inside the tunnel void.
  - *Modify when* changing tunnel shape, mask generation, point-in-polygon behavior, or tunnel coordinate convention.
- **`visualize_blocks.py`**: Block detection result visualization.
  - *Responsibilities:* 2D overview figures, block volume histograms, static 3D scatter plots, PyVista interactive block visualization.
  - *Modify only* for visualization of detected blocks. Do not place core block detection logic here.
- **`plot_3d_tunnel_fractures.py`**: DFN/tunnel topology visualization before block detection.
  - *Responsibilities:* Visualize fracture discs and tunnel geometry; filter fractures by tunnel intersection or inside/outside relationship; inspect topological relation between DFN and tunnel.
  - *Modify for* DFN/tunnel visualization, not for CCA.
- **`plot_2d_trace_map.py`**: 2D trace map viewer.
  - *Responsibilities:* Compute and visualize intersections between DFN and a selected X-plane; display trace lines on a face; support $P_{21}$ intensity checks.
  - *Modify for* trace-map viewing, trace clipping display, or $P_{21}$ visualization.

## 5. Goal-Driven Execution
- **Convert every coding request into a verifiable goal.**
  - *Example (Fix $P_{21}$ error):* Reproduce current $P_{21}$ calculation; identify whether error comes from deterministic, stochastic, trace clipping, or normalization; change only the responsible section; re-run and compare metrics.
  - *Example (Use 6-connectivity):* Locate CCA connectivity construction; ensure 6-neighbor structure is used; run small test grid to confirm diagonal-only contact does not merge blocks; check existing pipeline still runs.
  - *Example (Match traces between two faces):* Confirm coordinates and spacing; define candidate pairs; fit 3D plane from matched endpoints; reject geometrically unstable matches; export reconstructed planes.

## 6. Verification Before Completion
- **Do not claim success without verification.**
  - *Syntax-only change:* Run `python -m py_compile <changed_file.py>`
  - *Function-level change:* Run the smallest relevant test or script calling the changed function.
  - *Pipeline-level change:* Run the pipeline command with a small/known dataset and report command, exit status, key metrics, and warnings.
- **If the environment cannot run the code**, say so clearly and provide the exact verification command the user should run.

## 7. Geometry and Coordinate Safety
- **Be explicit with coordinate systems.**
  - **$x$**: tunnel advance direction.
  - **excavation face**: plane of approximately constant $x$.
  - **face-local coordinates**: commonly $[y, z]$.
  - **3D point**: commonly $[x, y, z]$.
  - **trace**: line segment created by intersection between fracture and excavation face.
  - **fracture**: plane, bounded disc, or reconstructed surface depending on context.
- **Never mix $[x, y, z]$ and $[y, z]$ silently.**
- **Make coordinate conversions obvious.** Use clear variables like `p0_yz`, `p1_yz`, `p0_xyz`, `trace_mid_yz`, `face_x`. Avoid ambiguous names like `p0`, `p1`, `coords`, or `points` when dimensionality matters.

## 8. DFN Reconstruction Rules
- **Do not overfit when reconstructing fractures from traces.**
- **Prefer a conservative sequence:**
  1. Use deterministic reconstruction *only* when trace evidence is strong.
  2. Use stochastic generation *only* for unexplained residual intensity.
  3. Prevent deterministic fractures from exceeding the target $P_{21}$ by themselves.
  4. Keep trace clipping and observation-window effects explicit.
  5. Separate observed traces, simulated traces, deterministic fractures, and stochastic fractures in logs.
- **Always report diagnostic outputs:** Observed/Simulated trace counts, Deterministic/Stochastic/Total fracture counts, Observed/Simulated $P_{21}$, $P_{21}$ error, orientation mismatch, trace length distribution mismatch, and total loss.
- **If deterministic fractures alone exceed target $P_{21}$, do not blindly add stochastic fractures.**

## 9. CCA and Block Detection Rules
- **Use conservative interpretation unless requested otherwise.**
- **Default connectivity:**
  - **6-connectivity** for rock voxel connectivity when the goal is to avoid merging blocks through edge or corner contact.
  - **26-connectivity** only when diagonal or corner-connected voxels should be considered physically connected.
- **Keep block filtering explicit in logs/CLI:** components touching tunnel surface, components *not* touching outer model boundary, separation by fracture/tunnel voxels, and any active volume/voxel thresholds.

## 10. GPU and Memory Safety
- **Do not assume GPU availability.** Keep CPU fallback behavior if it already exists.
- **Avoid unnecessary full-domain dense arrays.** Prefer bounding-box-limited operations.
- **Do not copy large arrays** between GPU and CPU repeatedly, and do not convert CuPy arrays to NumPy unless required for output/visualization.

## 11. HDF5 and Output Safety
- **Do not change HDF5 schema casually.** Expected fields:
  - `/fractures/centers`
  - `/fractures/normals`
  - `/fractures/radii`
  - `/fractures/set_id`
  - `/tunnel/*` and `/meta/*`
- **Before changing output structures**, identify downstream readers, preserve backward compatibility, and document new fields.
- **Always include metadata to reproduce the run** (branch/commit, grid resolution, connectivity, voxel size, input path, detected blocks, thresholds).

## 12. Visualization Rules
- **Keep visualization separate from core computation.**
- **Label axes with units and state coordinate systems (global XYZ vs. local YZ).**
- **Keep color meanings explicit** and report any active filters.
- **Use visualization to verify geometry, not to change geometry.**

## 13. Logging Rules
- **Pipeline logs must be concise but highly diagnostic.**
- **Avoid vague claims of success.** Always report before/after values and hard metrics.

## 14. No Speculative Refactoring
- **Do not restructure or rename modules/functions for aesthetic reasons.**
- **Do not introduce new dependencies** unless necessary. If introduced, explain why, where it is used, how to install, and the fallback behavior.

## 15. Antigravity Execution Style
- **Inspect only the relevant files.**
- **State assumptions briefly, make the smallest patch, and run verification.**
- **Provide a single complete copy-pasteable script** if a full script is asked, or a **minimal patch** if a patch is asked.

## 16. When to Stop and Ask
Stop and ask when any of the following are unclear and materially affect the implementation:
1. Whether trace data are 2D face-local or 3D global.
2. Whether fractures should be infinite planes or bounded discs.
3. Whether the matching target is one face, two faces, or multiple consecutive faces.
4. Whether CCA should use 6-connectivity or 26-connectivity.
5. Whether the validation target is $P_{21}$, $P_{32}$, trace count, trace length, orientation, or block count.
6. Whether stochastic fractures are allowed.
7. Whether changes should be made to the latest branch or the current local branch.

---

## 17. Final Response Format After Code Changes
After making code changes, always respond with:

### 변경 요약
- [파일 이름](file:///[절대 경로]): [수정 사항 요약]

### 검증
- **실행한 명령어**: `...`
- **결과**: `...` (또는 로컬에서 실행할 명령어 안내)

### 주의 사항
- `...`
