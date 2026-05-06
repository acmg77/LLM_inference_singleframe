# KuiperLLama Docker 一键启动脚本（Windows PowerShell）
# 用法: .\docker_run.ps1 [-Mode cpu|gpu] [-ModelType llama3|qwen2|qwen3] [-ModelDir <路径>]

param(
    [string]$Mode = "gpu",         # gpu 或 cpu
    [string]$ModelType = "qwen2",  # llama3, qwen2, qwen3
    [string]$ModelDir = ""         # 宿主机上的模型目录（可选）
)

$ProjectDir = $PSScriptRoot
$ImageName = "kuiper-llama:cuda"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " KuiperLLama Docker 启动脚本" -ForegroundColor Cyan
Write-Host " 模式: $Mode  |  模型类型: $ModelType" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 检查镜像是否存在
$imageExists = docker images -q $ImageName 2>$null
if (-not $imageExists) {
    Write-Host "`n[1/2] 镜像不存在，开始构建 $ImageName ..." -ForegroundColor Yellow
    docker build -t $ImageName -f "$ProjectDir\dockerfile.cuda" "$ProjectDir"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "镜像构建失败！" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n镜像 $ImageName 已存在，跳过构建。" -ForegroundColor Green
}

# 构建 docker run 参数
$GpuArgs = if ($Mode -eq "gpu") { "--gpus all" } else { "" }
$VolumeMounts = "-v `"${ProjectDir}:/workspaces/KuiperLLama`""
if ($ModelDir -ne "") {
    $VolumeMounts += " -v `"${ModelDir}:/models`""
}

Write-Host "`n[2/2] 启动容器（交互模式）..." -ForegroundColor Yellow
Write-Host "提示：进入容器后运行 './build_and_run.sh $ModelType' 编译项目" -ForegroundColor Gray

$DockerCmd = "docker run -it --rm $GpuArgs $VolumeMounts -w /workspaces/KuiperLLama $ImageName bash"
Write-Host "`n执行: $DockerCmd`n" -ForegroundColor DarkGray
Invoke-Expression $DockerCmd
