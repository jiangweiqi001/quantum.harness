# SCNet 超算 ED 工作流程简明教程

## 连接信息

```bash
# 密钥路径（仅首次或过期时更换）
KEY="/Users/dune/Downloads/mazhuijing_xh5.hpccube.com_RsaKeyExpireTime_2026-08-26_17-02-09.txt"

# SSH 连接
ssh -i "$KEY" -p 65061 mazhuijing@xh5.hpccube.com

# 快捷别名（可选，写入 ~/.ssh/config）
Host scnet
    HostName xh5.hpccube.com
    Port 65061
    User mazhuijing
    IdentityFile /Users/dune/Downloads/...RsaKey...txt
```

## 远端路径

```
/work/home/mazhuijing/ed-project/
├── .venv/                              # Python 3.11.5 venv（anaconda3/2023.09）
│   └── lib/python3.11/site-packages/   # QuSpin + 科学包
└── tracks/ed/hubbard-pump/             # 项目目录
    ├── run_ssh_ed.py                   # ED 计算脚本
    ├── test_ssh_ed.py                  # pytest 测试
    ├── ssh_ed.slurm                    # Slurm 提交脚本
    └── results/                        # 计算结果输出
```

## 标准操作流程

### 1. 本地编辑 → 远端同步

```bash
# 只同步项目目录，排除结果文件
rsync -av --exclude='results/' \
  -e "ssh -i $KEY -p 65061" \
  tracks/ed/hubbard-pump/ \
  mazhuijing@xh5.hpccube.com:/work/home/mazhuijing/ed-project/tracks/ed/hubbard-pump/
```

### 2. 远端快速验证（登录节点，不占用调度资源）

```bash
ssh -i "$KEY" -p 65061 mazhuijing@xh5.hpccube.com \
  "cd /work/home/mazhuijing/ed-project/tracks/ed/hubbard-pump && \
   module load anaconda3/2023.09 && \
   source /work/home/mazhuijing/ed-project/.venv/bin/activate && \
   pytest -q test_ssh_ed.py"
```

### 3. Slurm 脚本模板

```bash
#!/bin/bash
#SBATCH --job-name=<任务名>
#SBATCH --partition=xhacnormalb       # CPU 分区，128核/节点，~500GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<N核>         # 按需：1-128
#SBATCH --time=<HH:MM:SS>             # 上限 24:00:00

set -euo pipefail
module load anaconda3/2023.09
source /work/home/mazhuijing/ed-project/.venv/bin/activate
cd /work/home/mazhuijing/ed-project/tracks/ed/hubbard-pump

# 多线程 BLAS（利用多核）
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run_ssh_ed.py
```

### 4. 预检（不实际入队）

```bash
ssh -i "$KEY" -p 65061 mazhuijing@xh5.hpccube.com \
  "cd /work/home/mazhuijing/ed-project && sbatch --test-only tracks/ed/hubbard-pump/ssh_ed.slurm"
```

### 5. 提交任务

```bash
ssh -i "$KEY" -p 65061 mazhuijing@xh5.hpccube.com \
  "cd /work/home/mazhuijing/ed-project && sbatch tracks/ed/hubbard-pump/ssh_ed.slurm"
# 输出: Submitted batch job <JOBID>
```

### 6. 监控状态

```bash
# 查看作业状态
ssh -i "$KEY" -p 65061 mazhuijing@xh5.hpccube.com "squeue -j <JOBID>"

# 查看历史作业
ssh -i "$KEY" -p 65061 mazhuijing@xh5.hpccube.com \
  "sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS"

# 查看自己所有作业
ssh -i "$KEY" -p 65061 mazhuijing@xh5.hpccube.com "squeue -u mazhuijing"
```

### 7. 拉取结果

```bash
rsync -av \
  -e "ssh -i $KEY -p 65061" \
  mazhuijing@xh5.hpccube.com:/work/home/mazhuijing/ed-project/tracks/ed/hubbard-pump/results/ \
  tracks/ed/hubbard-pump/results/
```

### 8. 取消任务

```bash
ssh -i "$KEY" -p 65061 mazhuijing@xh5.hpccube.com "scancel <JOBID>"
```

## 可用分区

| 分区 | 类型 | 核/节点 | 内存/节点 | GPU |
|------|------|--------|----------|-----|
| `xhacnormalb` | CPU | 128 | ~500 GB | 无 |
| `xhhgnormal` | GPU | 64 | ~250 GB | NVIDIA×4 |
| `xhhgnormal01` | GPU | 64 | ~250 GB | NVIDIA×8 |
| `xhhgnormal02` | GPU | 64 | ~250 GB | V100×2 |

ED 使用 `xhacnormalb`。

## 一键提交脚本（本地运行）

```bash
#!/bin/bash
# submit.sh — 同步 + 预检 + 提交
KEY="/Users/dune/Downloads/mazhuijing_xh5.hpccube.com_RsaKeyExpireTime_2026-08-26_17-02-09.txt"
HOST="mazhuijing@xh5.hpccube.com"
PORT=65061

# 1. 同步代码
echo "=== Syncing ==="
rsync -av --exclude='results/' \
  -e "ssh -i $KEY -p $PORT" \
  tracks/ed/hubbard-pump/ \

  $HOST:/work/home/mazhuijing/ed-project/tracks/ed/hubbard-pump/

# 2. 预检
echo "=== Pre-check ==="
ssh -i "$KEY" -p "$PORT" "$HOST" \
  "cd /work/home/mazhuijing/ed-project && sbatch --test-only tracks/ed/hubbard-pump/ssh_ed.slurm"

# 3. 提交
echo "=== Submitting ==="
ssh -i "$KEY" -p "$PORT" "$HOST" \
  "cd /work/home/mazhuijing/ed-project && sbatch tracks/ed/hubbard-pump/ssh_ed.slurm"
```

## 性能提示

- `OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK` 让 numpy eigh() 使用所有请求的核
- 登录节点仅 2 核，不要在上面跑计算——只做验证和提交
- `--test-only` 预检免费，习惯性使用
- 结果 `results/` 不参与 rsync 上传，只拉取
