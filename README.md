# Audio Anti-Spoofing Thesis Cheatsheet

## Harian: Mulai Kerja

**Mulai kerja (setiap hari)**
```bash
# Login (password: 4 spasi)
ssh user3@100.111.211.254

# Masuk ke folder kerja
cd /home/user3/Developer/Nemeziz/audio-antispoofing-thesis

# Aktivasi environment
conda activate antispoofing

# Masuk ke sesi tmux untuk lihat progress training
tmux attach -t thesis_training
```

**Keluar dengan aman**
- `Ctrl+B lalu D` : Detach dari tmux (training TETAP jalan).
- `exit` : Keluar dari SSH (training tetap jalan selama tmux aktif).

**Pantau training (tanpa buka tmux)**
```bash
# GPU utilization & VRAM (update tiap 5 detik)
watch -n 5 nvidia-smi

# Live log training (ganti nama model sesuai yang jalan)
tail -f experiments/run_logs/se_rawformer_schema_b.log

# W&B dashboard (pantau dari browser/HP kapan saja)
# https://wandb.ai/itera/audio-antispoofing-thesis
```

---

## Manajemen Sesi: tmux

**Commands**
```bash
tmux new-session -s thesis_training   # Buat sesi baru (hanya jika belum ada)
tmux ls                               # Lihat semua sesi berjalan
tmux attach -t thesis_training        # Masuk ke sesi
tmux kill-session -t thesis_training  # Hapus sesi (HANYA setelah training selesai!)
```

**Keyboard Shortcuts (di dalam tmux)**
- `Ctrl+B → D` : Detach (aman)
- `Ctrl+B → lalu PgUp` : Scroll ke atas
- `Q` : Keluar scroll
- `Ctrl+B → C` : Buat window baru
- `Ctrl+B → N` : Pindah window
- `Ctrl+B → "` : Split horizontal
- `Ctrl+C` : Stop training (paksa)

---

## Training: Urutan Eksekusi

### Schema B — Utama (RawBoost)
```bash
# 1/8
python main.py --mode train --model se_rawformer --config configs/schema_b.yaml

# 2/8
python main.py --mode train --model rawtfnet_32 --config configs/schema_b.yaml

# 3/8
python main.py --mode train --model rawtfnet_16 --config configs/schema_b.yaml

# 4/8
python main.py --mode train --model aasist --config configs/schema_b.yaml
```

### Schema A — Ablation (Tanpa Augmentasi)
```bash
# 5/8
python main.py --mode train --model se_rawformer --config configs/schema_a.yaml

# 6/8
python main.py --mode train --model rawtfnet_32 --config configs/schema_a.yaml

# 7/8
python main.py --mode train --model rawtfnet_16 --config configs/schema_a.yaml

# 8/8
python main.py --mode train --model aasist --config configs/schema_a.yaml
```

### Evaluasi & Resume
```bash
# Resume dari checkpoint (ganti path sesuai fold)
python main.py --mode train --model se_rawformer --config configs/schema_b.yaml --resume experiments/se_rawformer/schema_b/fold_0/best.pt

# Evaluasi akhir setelah semua fold selesai
python main.py --mode eval --model se_rawformer --config configs/schema_b.yaml

# Verifikasi GPU + model forward pass
python main.py --mode verify
```

---

## Debug & Checking

**Kapasitas Komputer**
```bash
# Status GPU & VRAM
nvidia-smi

# Live update GPU
watch -n 2 nvidia-smi

# Cek sisa ukuran hardisk
df -h /home/user3

# Ukuran folder output experiments
du -sh experiments/
```

**Smoke test Dataloader**
```bash
python -c "
from src.dataloader import KFoldManager, InTheWildDataset
import torch
kfold = KFoldManager('data/ASVspoof2019_LA', n_splits=5, seed=42)
tr, va = kfold.get_fold(0, augment=False, config={})
x,y,f = tr[0]; assert x.shape==torch.Size([64000]); print('KFold OK:', x.shape, y)
itw = InTheWildDataset('data/in_the_wild')
x,y,f = itw[0]; assert x.shape==torch.Size([64000]); print('ITW OK:', x.shape, y, len(itw))"
```

**Cari File Checkpoint Terakhir**
```bash
# Semua checkpoint sorted:
find experiments/ -name "*.pt" | sort

# 10 Checkpoint terbaru:
find experiments/ -name "*.pt" | xargs ls -lht | head -10
```

**Troubleshooting Cepat**
- **GPU OOM:** Kurangi batch_size di config yaml
- **Loss = NaN:** Coba lr=1e-5, cek label dataloader
- **EER tidak turun:** Cek label konvensi (bonafide=1)
- **Training hang:** `pgrep -a python` → `kill -SIGTERM [PID]`
- **W&B offline:** `export WANDB_MODE=offline` lalu sync

---

## Spesifikasi & Parameter

**Hardware**
- PC: `user3@100.111.211.254` 
- GPU: `RTX 5070 12GB`
- CPU Core: `12`
- RAM: `32GB`
- CUDA: `12.8`

**Training Config (Update: High-Performance)**
- Workers: `4` (Formula: 12 cores - 2)
- Batch Size / Accumulator: `48 x 2 = eff 96`
- Optimizer: `Adam lr=1e-4`
- Max Epochs: `100` (Eary Stop 10)
- K-fold: `5 split (Seed 42, deterministic=false)`
- Precision: `bf16 + GradScaler`
- Input: `64.000 (16kHz, 4 detik)`

**Dataset**
- ASVspoof train+dev: `50.224` (Imbalance 1 : 8.8)
- ASVspoof eval: `71.237`
- In-the-Wild: `31.779`

**Path**
- Project: `/home/user3/Developer/Nemeziz/audio-antispoofing-thesis`
- Configs: `configs/schema_b.yaml` dan `configs/schema_a.yaml`
- ASVspoof: `data/ASVspoof2019_LA/`
- In-the-Wild: `data/in_the_wild/release_in_the_wild/`
- Output Weights: `experiments/`
