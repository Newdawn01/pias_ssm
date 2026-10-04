# Processed Data

Place the processed PyTorch files in this directory:

- `cwru_workbuddy_10way.pt`
- `ottawa_workbuddy.pt`

The JSON configurations in `configs/` use these relative paths. The processed files are
not shipped with the code; regenerate them with `scripts/preprocess_cwru.py` and
`scripts/preprocess_ottawa.py` (see the top-level `README.md`).

Each file is a `torch.save` dictionary with the following keys:

| Key | Shape | Meaning |
|-----|-------|---------|
| `samples` | `[N, 2, L]` | channel 0 = vibration, channel 1 = normalised instantaneous speed |
| `fault_labels` | `[N]` | integer fault class |
| `condition_labels` | `[N]` | integer operating condition |
| `file_ids` | `[N]` | source recording / temporal block, used for the leakage-safe split |
| `instantaneous_rpm` | `[N, L]` | instantaneous shaft speed in rpm |
| `fault_freq_prior` | `[N, 4, L]` | BPFI / BPFO / BSF / FTF in Hz, per time step |
| `target_fs` | scalar | sampling rate of the processed signal (CWRU 12000 Hz, Ottawa 2000 Hz) |

Optional keys written by the preprocessing scripts:

| Key | Meaning |
|-----|---------|
| `window_len` | window length in samples |
| `stride` | hop between consecutive windows |
| `block_size` | CWRU only: number of windows grouped into one `file_id` |
| `file_id_map` | Ottawa only: `file_id` to source `.mat` filename |
| `split_unit` | human-readable description of the leakage-safe split unit |
