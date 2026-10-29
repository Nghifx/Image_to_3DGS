# GSplat Studio - one-time setup on a new machine. Run via setup.bat (double-click).
#   -NoHoudini  skip registering the Houdini package (used when testing a copy)
param([switch]$NoHoudini)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

Step "Python 3.13"
$py = $null
foreach ($cand in @(@("py", "-3.13"), @("python"))) {
    try {
        $v = & $cand[0] $cand[1..9] --version 2>&1
        if ("$v" -match "Python 3\.13") { $py = $cand; break }
    } catch {}
}
if (-not $py) {
    Write-Host "Can cai Python 3.13 (python.org, tick 'Add to PATH') roi chay lai setup.bat" -ForegroundColor Red
    exit 1
}
Write-Host "dung: $($py -join ' ')"

Step "Moi truong Python cho cac buoc tach nen + COLMAP (tools\venv)"
$venvPy = Join-Path $root "tools\venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    & $py[0] $py[1..9] -m venv (Join-Path $root "tools\venv")
}
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Host "Cai goi Python loi" -ForegroundColor Red; exit 1 }

Step "Model tach nen (rembg)"
$model = Join-Path $root "tools\models\isnet-general-use.onnx"
$target = Join-Path $env:USERPROFILE ".rembg\models\isnet-general-use"
if ((Test-Path $model) -and -not (Test-Path (Join-Path $target "isnet-general-use.onnx"))) {
    New-Item -ItemType Directory -Force $target | Out-Null
    Copy-Item $model $target
}
Write-Host "ok"

Step "ffmpeg"
$ffmpeg = Join-Path $root "tools\ffmpeg\ffmpeg.exe"
if (Test-Path $ffmpeg) { Write-Host "ok: $ffmpeg" } else { Write-Host "thieu tools\ffmpeg\ffmpeg.exe" -ForegroundColor Yellow }

Step "Anh cho cac bo COLMAP co san (lien ket toi data\<ten>\images, khong ton them dia)"
foreach ($ds in Get-ChildItem (Join-Path $root "colmap") -Directory -ErrorAction SilentlyContinue) {
    $src = Join-Path $root "data\$($ds.Name)\images"
    $dst = Join-Path $ds.FullName "images"
    if ($ds.Name -eq "_work" -or -not (Test-Path $src) -or (Test-Path $dst)) { continue }
    New-Item -ItemType Directory -Force $dst | Out-Null
    foreach ($f in Get-ChildItem $src -File) {
        try { New-Item -ItemType HardLink -Path (Join-Path $dst $f.Name) -Target $f.FullName | Out-Null }
        catch { Copy-Item $f.FullName $dst }
    }
    Write-Host "ok: colmap\$($ds.Name)\images"
}

if (-not $NoHoudini) {
    Step "Dang ky plugin GSplat Studio voi Houdini 22"
    $docs = [Environment]::GetFolderPath("MyDocuments")
    $homes = @(Get-ChildItem $docs -Directory -Filter "houdini22.*" -ErrorAction SilentlyContinue)
    if ($homes.Count -eq 0) { $homes = @(New-Item -ItemType Directory -Force (Join-Path $docs "houdini22.0")) }
    $fwd = $root -replace "\\", "/"
    $json = @"
{
    "enable": true,
    "env": [
        {"GSPLAT_STUDIO": "$fwd/houdini/gsplat_studio"},
        {"GSPLAT_ROOT": "$fwd"}
    ],
    "path": "`$GSPLAT_STUDIO"
}
"@
    foreach ($h in $homes) {
        $pk = Join-Path $h.FullName "packages"
        New-Item -ItemType Directory -Force $pk | Out-Null
        Set-Content -Path (Join-Path $pk "gsplat_studio.json") -Value $json -Encoding utf8
        Write-Host "ok: $pk\gsplat_studio.json"
    }
}

Step "Kiem tra"
& $venvPy -c "import pycolmap, rembg, cv2, numpy; print('pycolmap', pycolmap.__version__, '| opencv', cv2.__version__)"
$hython = Get-ChildItem "C:\Program Files\Side Effects Software\Houdini 22*\bin\hython.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($hython) { Write-Host "Houdini: $($hython.FullName)" } else { Write-Host "Chua thay Houdini 22 - cai Houdini 22 roi mo panel GSplat Studio" -ForegroundColor Yellow }
Write-Host "`nXONG. Mo Houdini 22 > + > New Pane Tab Type > GSplat Studio" -ForegroundColor Green
