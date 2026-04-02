# QExtract-Infer NCU Profiling 脚本
# 使用 NVIDIA Nsight Compute 抓取算子的显存吞吐量数据
# 用法: powershell -File benchmarks/profile_ncu.ps1

$ErrorActionPreference = "Stop"

# ── 配置 ──
$PYTHON = "python"
$NCU = "ncu"  # 确保 ncu 在 PATH 中 (CUDA Toolkit 安装后自带)

$OUTPUT_DIR = "benchmarks/ncu_reports"
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

Write-Host "======================================"
Write-Host "  QExtract-Infer NCU Profiling"
Write-Host "======================================"

# ── Profile RMSNorm ──
Write-Host "`n[1/3] Profiling RMSNorm..."
& $NCU --set full `
    --target-processes all `
    --kernel-name "fused_rmsnorm" `
    --export "$OUTPUT_DIR/rmsnorm_profile" `
    $PYTHON -c "
import torch
from qextract._C import fused_rmsnorm
x = torch.randn(1, 2560, dtype=torch.float16, device='cuda')
w = torch.randn(2560, dtype=torch.float16, device='cuda')
for _ in range(10):
    fused_rmsnorm(x, w, 1e-6)
torch.cuda.synchronize()
"

# ── Profile SwiGLU ──
Write-Host "`n[2/3] Profiling SwiGLU..."
& $NCU --set full `
    --target-processes all `
    --kernel-name "fused_swiglu" `
    --export "$OUTPUT_DIR/swiglu_profile" `
    $PYTHON -c "
import torch
from qextract._C import fused_swiglu
g = torch.randn(1, 9728, dtype=torch.float16, device='cuda')
u = torch.randn(1, 9728, dtype=torch.float16, device='cuda')
for _ in range(10):
    fused_swiglu(g, u)
torch.cuda.synchronize()
"

# ── Profile W4A16 GEMV ──
Write-Host "`n[3/3] Profiling W4A16 GEMV..."
& $NCU --set full `
    --target-processes all `
    --kernel-name "w4a16" `
    --export "$OUTPUT_DIR/w4a16_gemv_profile" `
    $PYTHON -c "
import torch
from qextract._C import w4a16_gemv
hidden = 2560
out_feat = 2560
x = torch.randn(hidden, dtype=torch.float16, device='cuda')
qw = torch.randint(0, 2**31, (hidden//8, out_feat), dtype=torch.int32, device='cuda')
s = torch.randn(hidden//128, out_feat, dtype=torch.float16, device='cuda') * 0.1
z = torch.randn(hidden//128, out_feat, dtype=torch.float16, device='cuda')
for _ in range(10):
    w4a16_gemv(x, qw, s, z, 128)
torch.cuda.synchronize()
"

Write-Host "`n✅ NCU 报告已保存到 $OUTPUT_DIR/"
Write-Host "使用 Nsight Compute GUI 打开 .ncu-rep 文件查看火焰图"
