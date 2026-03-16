# Audio Anti-Spoofing Comparative Study

<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--font-mono,monospace)}
.cs{padding:1rem 0;max-width:100%}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:1rem}
.tab{padding:6px 14px;border-radius:var(--border-radius-md);border:0.5px solid var(--color-border-secondary);background:var(--color-background-primary);color:var(--color-text-secondary);cursor:pointer;font-size:13px;font-family:var(--font-sans)}
.tab.active{background:var(--color-background-secondary);color:var(--color-text-primary);border-color:var(--color-border-primary);font-weight:500}
.section{display:none}
.section.active{display:block}
.card{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:1rem 1.25rem;margin-bottom:12px}
.card-title{font-size:12px;font-weight:500;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-family:var(--font-sans)}
.cmd{background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:8px 12px;margin-bottom:6px;display:flex;align-items:flex-start;gap:10px}
.cmd-text{font-size:13px;color:var(--color-text-primary);flex:1;white-space:pre-wrap;line-height:1.5}
.cmd-note{font-size:11px;color:var(--color-text-secondary);margin-top:3px;font-family:var(--font-sans)}
.copy-btn{background:none;border:0.5px solid var(--color-border-secondary);border-radius:var(--border-radius-md);padding:3px 8px;font-size:11px;color:var(--color-text-secondary);cursor:pointer;white-space:nowrap;font-family:var(--font-sans);flex-shrink:0}
.copy-btn:hover{background:var(--color-background-secondary)}
.copy-btn.copied{color:var(--color-text-success);border-color:var(--color-border-success)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.badge{display:inline-block;padding:2px 8px;border-radius:var(--border-radius-md);font-size:11px;font-family:var(--font-sans)}
.badge-ok{background:var(--color-background-success);color:var(--color-text-success)}
.badge-warn{background:var(--color-background-warning);color:var(--color-text-warning)}
.badge-info{background:var(--color-background-info);color:var(--color-text-info)}
.row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:0.5px solid var(--color-border-tertiary);font-size:13px;font-family:var(--font-sans)}
.row:last-child{border-bottom:none}
.row-label{color:var(--color-text-secondary)}
.row-val{color:var(--color-text-primary);font-weight:500;font-family:var(--font-mono)}
.section-label{font-size:11px;font-weight:500;color:var(--color-text-tertiary);text-transform:uppercase;letter-spacing:.06em;margin:14px 0 6px;font-family:var(--font-sans)}
</style>

<div class="cs">
<div class="tabs">
  <button class="tab active" onclick="switchTab('daily')">Harian</button>
  <button class="tab" onclick="switchTab('tmux')">tmux</button>
  <button class="tab" onclick="switchTab('training')">Training</button>
  <button class="tab" onclick="switchTab('debug')">Debug & Cek</button>
  <button class="tab" onclick="switchTab('specs')">Spesifikasi</button>
</div>

<div id="daily" class="section active">
  <div class="card">
    <div class="card-title">Mulai kerja (setiap hari)</div>
    <div class="cmd"><div><div class="cmd-text">ssh user3@100.111.211.254</div><div class="cmd-note">password: 4 spasi</div></div><button class="copy-btn" onclick="cp(this,'ssh user3@100.111.211.254')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">tmux attach -t thesis_training</div><div class="cmd-note">lihat progress training</div></div><button class="copy-btn" onclick="cp(this,'tmux attach -t thesis_training')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">cd /home/user3/Developer/Nemeziz/audio-antispoofing-thesis</div></div><button class="copy-btn" onclick="cp(this,'cd /home/user3/Developer/Nemeziz/audio-antispoofing-thesis')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">conda activate antispoofing</div></div><button class="copy-btn" onclick="cp(this,'conda activate antispoofing')">copy</button></div>
  </div>
  <div class="card">
    <div class="card-title">Keluar dengan aman</div>
    <div class="cmd"><div><div class="cmd-text">Ctrl+B  lalu  D</div><div class="cmd-note">detach dari tmux — training TETAP jalan</div></div></div>
    <div class="cmd"><div><div class="cmd-text">exit</div><div class="cmd-note">keluar dari SSH — training tetap jalan selama tmux aktif</div></div></div>
  </div>
  <div class="card">
    <div class="card-title">Pantau training (tanpa buka tmux)</div>
    <div class="cmd"><div><div class="cmd-text">watch -n 5 nvidia-smi</div><div class="cmd-note">GPU utilization & VRAM — update tiap 5 detik</div></div><button class="copy-btn" onclick="cp(this,'watch -n 5 nvidia-smi')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">tail -f experiments/run_logs/se_rawformer_schema_b.log</div><div class="cmd-note">live log training — ganti nama model sesuai yang jalan</div></div><button class="copy-btn" onclick="cp(this,'tail -f experiments/run_logs/se_rawformer_schema_b.log')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">https://wandb.ai/itera/audio-antispoofing-thesis</div><div class="cmd-note">pantau dari browser / HP kapan saja</div></div><button class="copy-btn" onclick="cp(this,'https://wandb.ai/itera/audio-antispoofing-thesis')">copy</button></div>
  </div>
</div>

<div id="tmux" class="section">
  <div class="card">
    <div class="card-title">Manajemen sesi</div>
    <div class="cmd"><div><div class="cmd-text">tmux new-session -s thesis_training</div><div class="cmd-note">buat sesi baru (hanya jika belum ada)</div></div><button class="copy-btn" onclick="cp(this,'tmux new-session -s thesis_training')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">tmux ls</div><div class="cmd-note">lihat semua sesi yang berjalan</div></div><button class="copy-btn" onclick="cp(this,'tmux ls')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">tmux attach -t thesis_training</div><div class="cmd-note">masuk ke sesi</div></div><button class="copy-btn" onclick="cp(this,'tmux attach -t thesis_training')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">tmux kill-session -t thesis_training</div><div class="cmd-note">hapus sesi (hanya setelah training selesai!)</div></div><button class="copy-btn" onclick="cp(this,'tmux kill-session -t thesis_training')">copy</button></div>
  </div>
  <div class="card">
    <div class="card-title">Keyboard shortcuts (di dalam tmux)</div>
    <div class="row"><span class="row-label">Detach (aman)</span><span class="row-val">Ctrl+B → D</span></div>
    <div class="row"><span class="row-label">Scroll ke atas</span><span class="row-val">Ctrl+B → [  lalu  PgUp</span></div>
    <div class="row"><span class="row-label">Keluar scroll</span><span class="row-val">Q</span></div>
    <div class="row"><span class="row-label">Buat window baru</span><span class="row-val">Ctrl+B → C</span></div>
    <div class="row"><span class="row-label">Pindah window</span><span class="row-val">Ctrl+B → N</span></div>
    <div class="row"><span class="row-label">Split horizontal</span><span class="row-val">Ctrl+B → "</span></div>
    <div class="row"><span class="row-label">Stop training (paksa)</span><span class="row-val">Ctrl+C</span></div>
  </div>
</div>

<div id="training" class="section">
  <div class="card">
    <div class="card-title">Urutan 8 eksperimen</div>
    <div class="section-label">Schema B — utama</div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode train --model se_rawformer --config configs/schema_b.yaml</div><div class="cmd-note">1/8 — sedang jalan sekarang</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode train --model se_rawformer --config configs/schema_b.yaml')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode train --model rawtfnet_32 --config configs/schema_b.yaml</div><div class="cmd-note">2/8</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode train --model rawtfnet_32 --config configs/schema_b.yaml')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode train --model rawtfnet_16 --config configs/schema_b.yaml</div><div class="cmd-note">3/8</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode train --model rawtfnet_16 --config configs/schema_b.yaml')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode train --model aasist --config configs/schema_b.yaml</div><div class="cmd-note">4/8</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode train --model aasist --config configs/schema_b.yaml')">copy</button></div>
    <div class="section-label">Schema A — ablation</div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode train --model se_rawformer --config configs/schema_a.yaml</div><div class="cmd-note">5/8</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode train --model se_rawformer --config configs/schema_a.yaml')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode train --model rawtfnet_32 --config configs/schema_a.yaml</div><div class="cmd-note">6/8</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode train --model rawtfnet_32 --config configs/schema_a.yaml')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode train --model rawtfnet_16 --config configs/schema_a.yaml</div><div class="cmd-note">7/8</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode train --model rawtfnet_16 --config configs/schema_a.yaml')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode train --model aasist --config configs/schema_a.yaml</div><div class="cmd-note">8/8</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode train --model aasist --config configs/schema_a.yaml')">copy</button></div>
  </div>
  <div class="card">
    <div class="card-title">Resume & evaluasi</div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode train --model se_rawformer --config configs/schema_b.yaml --resume experiments/se_rawformer/schema_b/fold_0/best.pt</div><div class="cmd-note">resume dari checkpoint — ganti path sesuai fold</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode train --model se_rawformer --config configs/schema_b.yaml --resume experiments/se_rawformer/schema_b/fold_0/best.pt')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode eval --model se_rawformer --config configs/schema_b.yaml</div><div class="cmd-note">evaluasi akhir setelah semua fold selesai</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode eval --model se_rawformer --config configs/schema_b.yaml')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">python main.py --mode verify</div><div class="cmd-note">verifikasi GPU + model forward pass</div></div><button class="copy-btn" onclick="cp(this,'python main.py --mode verify')">copy</button></div>
  </div>
</div>

<div id="debug" class="section">
  <div class="grid2">
    <div class="card">
      <div class="card-title">Cek GPU</div>
      <div class="cmd"><div><div class="cmd-text">nvidia-smi</div><div class="cmd-note">status GPU, VRAM, proses</div></div><button class="copy-btn" onclick="cp(this,'nvidia-smi')">copy</button></div>
      <div class="cmd"><div><div class="cmd-text">watch -n 2 nvidia-smi</div><div class="cmd-note">live update tiap 2 detik</div></div><button class="copy-btn" onclick="cp(this,'watch -n 2 nvidia-smi')">copy</button></div>
    </div>
    <div class="card">
      <div class="card-title">Cek disk & dataset</div>
      <div class="cmd"><div><div class="cmd-text">df -h /home/user3</div><div class="cmd-note">sisa disk</div></div><button class="copy-btn" onclick="cp(this,'df -h /home/user3')">copy</button></div>
      <div class="cmd"><div><div class="cmd-text">du -sh experiments/</div><div class="cmd-note">ukuran folder experiments</div></div><button class="copy-btn" onclick="cp(this,'du -sh experiments/')">copy</button></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Smoke test dataloader</div>
    <div class="cmd"><div><div class="cmd-text">python -c "
from src.dataloader import KFoldManager, InTheWildDataset
import torch
kfold = KFoldManager('data/ASVspoof2019_LA', n_splits=5, seed=42)
tr, va = kfold.get_fold(0, augment=False, config={})
x,y,f = tr[0]; assert x.shape==torch.Size([64000]); print('KFold OK:', x.shape, y)
itw = InTheWildDataset('data/in_the_wild')
x,y,f = itw[0]; assert x.shape==torch.Size([64000]); print('ITW OK:', x.shape, y, len(itw))"</div></div><button class="copy-btn" onclick="cp(this,`python -c \"\nfrom src.dataloader import KFoldManager, InTheWildDataset\nimport torch\nkfold = KFoldManager('data/ASVspoof2019_LA', n_splits=5, seed=42)\ntr, va = kfold.get_fold(0, augment=False, config={})\nx,y,f = tr[0]; assert x.shape==torch.Size([64000]); print('KFold OK:', x.shape, y)\nitw = InTheWildDataset('data/in_the_wild')\nx,y,f = itw[0]; assert x.shape==torch.Size([64000]); print('ITW OK:', x.shape, y, len(itw))\"`)">copy</button></div>
  </div>
  <div class="card">
    <div class="card-title">Cek checkpoint yang ada</div>
    <div class="cmd"><div><div class="cmd-text">find experiments/ -name &quot;*.pt&quot; | sort</div><div class="cmd-note">semua checkpoint</div></div><button class="copy-btn" onclick="cp(this,'find experiments/ -name &quot;*.pt&quot; | sort')">copy</button></div>
    <div class="cmd"><div><div class="cmd-text">find experiments/ -name &quot;*.pt&quot; | xargs ls -lht | head -10</div><div class="cmd-note">checkpoint terbaru</div></div><button class="copy-btn" onclick="cp(this,'find experiments/ -name &quot;*.pt&quot; | xargs ls -lht | head -10')">copy</button></div>
  </div>
  <div class="card">
    <div class="card-title">Jika ada masalah</div>
    <div class="row"><span class="row-label">GPU OOM</span><span class="row-val" style="font-size:11px">kurangi batch_size di config yaml</span></div>
    <div class="row"><span class="row-label">Loss = NaN</span><span class="row-val" style="font-size:11px">coba lr=1e-5, cek label dataloader</span></div>
    <div class="row"><span class="row-label">EER tidak turun</span><span class="row-val" style="font-size:11px">cek label konvensi (bonafide=1)</span></div>
    <div class="row"><span class="row-label">Training hang</span><span class="row-val" style="font-size:11px">pgrep -a python → kill -SIGTERM [PID]</span></div>
    <div class="row"><span class="row-label">wandb offline</span><span class="row-val" style="font-size:11px">export WANDB_MODE=offline lalu sync</span></div>
  </div>
</div>

<div id="specs" class="section">
  <div class="grid2">
    <div class="card">
      <div class="card-title">Lingkungan</div>
      <div class="row"><span class="row-label">PC</span><span class="row-val">user3@100.111.211.254</span></div>
      <div class="row"><span class="row-label">Password</span><span class="row-val">[4 spasi]</span></div>
      <div class="row"><span class="row-label">GPU</span><span class="row-val">RTX 5070 12GB</span></div>
      <div class="row"><span class="row-label">CUDA</span><span class="row-val">12.8</span></div>
      <div class="row"><span class="row-label">PyTorch</span><span class="row-val">2.12 nightly</span></div>
      <div class="row"><span class="row-label">conda env</span><span class="row-val">antispoofing</span></div>
      <div class="row"><span class="row-label">Disk free</span><span class="row-val">~1.5 TB</span></div>
    </div>
    <div class="card">
      <div class="card-title">Dataset</div>
      <div class="row"><span class="row-label">ASVspoof train+dev</span><span class="row-val">50.224</span></div>
      <div class="row"><span class="row-label">ASVspoof eval</span><span class="row-val">71.237</span></div>
      <div class="row"><span class="row-label">In-the-Wild</span><span class="row-val">31.779</span></div>
      <div class="row"><span class="row-label">Imbalance</span><span class="row-val">1 : 8.8</span></div>
      <div class="row"><span class="row-label">Input length</span><span class="row-val">64.000 samples (4s)</span></div>
      <div class="row"><span class="row-label">Sample rate</span><span class="row-val">16.000 Hz</span></div>
    </div>
    <div class="card">
      <div class="card-title">Model specs</div>
      <div class="row"><span class="row-label">AASIST</span><span class="row-val">0.30M — 8.9G MACs</span></div>
      <div class="row"><span class="row-label">SE-Rawformer</span><span class="row-val">0.37M — 6.1G MACs</span></div>
      <div class="row"><span class="row-label">RawTFNet-32</span><span class="row-val">0.17M — 5.4G MACs</span></div>
      <div class="row"><span class="row-label">RawTFNet-16</span><span class="row-val">0.07M — 2.9G MACs</span></div>
    </div>
    <div class="card">
      <div class="card-title">Training config</div>
      <div class="row"><span class="row-label">Optimizer</span><span class="row-val">Adam lr=1e-4</span></div>
      <div class="row"><span class="row-label">Batch / accum</span><span class="row-val">128 × 1 = eff 128</span></div>
      <div class="row"><span class="row-label">Max epochs</span><span class="row-val">100</span></div>
      <div class="row"><span class="row-label">Early stop</span><span class="row-val">patience=10</span></div>
      <div class="row"><span class="row-label">K-Fold</span><span class="row-val">5 splits, seed=42</span></div>
      <div class="row"><span class="row-label">Precision</span><span class="row-val">bf16 + GradScaler</span></div>
      <div class="row"><span class="row-label">Loss</span><span class="row-val">Weighted CrossEntropy</span></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Path penting</div>
    <div class="row"><span class="row-label">Project</span><span class="row-val">/home/user3/Developer/Nemeziz/audio-antispoofing-thesis</span></div>
    <div class="row"><span class="row-label">ASVspoof</span><span class="row-val">data/ASVspoof2019_LA/</span></div>
    <div class="row"><span class="row-label">In-the-Wild</span><span class="row-val">data/in_the_wild/release_in_the_wild/</span></div>
    <div class="row"><span class="row-label">Experiments</span><span class="row-val">experiments/</span></div>
    <div class="row"><span class="row-label">Config B</span><span class="row-val">configs/schema_b.yaml</span></div>
    <div class="row"><span class="row-label">Config A</span><span class="row-val">configs/schema_a.yaml</span></div>
    <div class="row"><span class="row-label">wandb</span><span class="row-val">wandb.ai/itera/audio-antispoofing-thesis</span></div>
  </div>
</div>
</div>

<script>
function switchTab(id){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(id).classList.add('active');
}
function cp(btn,text){
  navigator.clipboard.writeText(text).then(()=>{
    btn.textContent='copied';btn.classList.add('copied');
    setTimeout(()=>{btn.textContent='copy';btn.classList.remove('copied')},1500);
  });
}
</script>
