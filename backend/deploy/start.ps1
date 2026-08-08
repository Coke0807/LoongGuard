<#
.SYNOPSIS
    LoongGuard 后端一键启动脚本（Windows 开发/测试环境）

.DESCRIPTION
    自动加载同目录 .env、启动后端主循环、崩溃自动重启。
    需在 PowerShell 7 (pwsh) 下运行：
        pwsh ./start.ps1

    退出码约定（配合自动重启策略）：
        0/1  = 正常退出或配置/端口错误（致命，不重启，避免死循环）
        其他 = 崩溃（5 秒后自动重启）
#>

$ErrorActionPreference = "Stop"

# 切换工作目录到脚本所在目录（backend/）
$backendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $backendDir

# ── 1. 加载 .env（若缺失 SM4 密钥，config 校验会在启动时拦截并报错）──
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $key, $value = $line -split "=", 2
            $key = $key.Trim()
            # 去掉行内注释（# 之后），保留值内 # 前内容
            $value = ($value -split "#")[0].Trim()
            if ($key -and -not [Environment]::GetEnvironmentVariable($key)) {
                [Environment]::SetEnvironmentVariable($key, $value, "Process")
            }
        }
    }
    Write-Host "[env] 已加载 .env" -ForegroundColor DarkGreen
} else {
    Write-Host "[env] 未找到 .env，将使用默认配置" -ForegroundColor Yellow
}

# ── 2. 选择 Python 解释器（优先项目虚拟环境）──
$py = $null
if (Test-Path ".venv\Scripts\python.exe")       { $py = ".venv\Scripts\python.exe" }
elseif (Test-Path "venv\Scripts\python.exe")    { $py = "venv\Scripts\python.exe" }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" }
else { Write-Error "未找到 Python。请先创建虚拟环境（python -m venv .venv）并安装依赖。"; exit 1 }

Write-Host "使用解释器: $py" -ForegroundColor DarkCyan
Write-Host "[run] 后端端口: $env:LG_API_PORT (默认 8080)" -ForegroundColor DarkCyan

# ── 3. 崩溃自动重启循环 ──
while ($true) {
    Write-Host "[run] 启动 LoongGuard 后端..." -ForegroundColor Cyan
    & $py -m src.pipeline
    $code = $LASTEXITCODE

    if ($code -eq 0 -or $code -eq 1) {
        Write-Host "[exit] 后端退出 (code=$code)，属正常/配置错误，不再自动重启" -ForegroundColor DarkGray
        break
    }
    Write-Host "[restart] 后端崩溃 (code=$code)，5 秒后自动重启..." -ForegroundColor Red
    Start-Sleep -Seconds 5
}