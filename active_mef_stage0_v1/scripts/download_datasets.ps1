param(
    [ValidateSet("kalantari", "sice", "real-hdrv", "deephdrvideo", "all")]
    [string[]]$Dataset = @("all"),

    [string]$Root = "data/raw",

    [switch]$NoExtract,
    [switch]$Force,
    [switch]$ListOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RootPath = if ([System.IO.Path]::IsPathRooted($Root)) {
    $Root
} else {
    Join-Path $RepoRoot $Root
}

function Resolve-DatasetSelection {
    param([string[]]$Names)

    if ($Names -contains "all") {
        return @("kalantari", "sice", "real-hdrv", "deephdrvideo")
    }
    return $Names
}

function Ensure-Directory {
    param([string]$Path)

    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Download-Url {
    param(
        [string]$Url,
        [string]$OutFile,
        [int64]$ExpectedBytes = 0
    )

    Ensure-Directory (Split-Path -Parent $OutFile)

    if ((Test-Path $OutFile) -and $Force) {
        Remove-Item -LiteralPath $OutFile -Force
    }

    if ((Test-Path $OutFile) -and -not $Force) {
        $currentBytes = (Get-Item -LiteralPath $OutFile).Length
        if (($ExpectedBytes -gt 0) -and ($currentBytes -eq $ExpectedBytes)) {
            Write-Host "Skip complete file: $OutFile"
            return
        }
        if ($ExpectedBytes -gt 0) {
            Write-Host "Resume partial file: $OutFile ($currentBytes / $ExpectedBytes bytes)"
        } else {
            Write-Host "Resume existing file: $OutFile ($currentBytes bytes)"
        }
    }

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $curl) {
        throw "curl.exe not found. Install curl or use a recent Windows build."
    }

    Write-Host "Downloading: $Url"
    & $curl.Source -L -C - --fail --retry 5 --retry-delay 5 -o $OutFile $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Url"
    }

    if ($ExpectedBytes -gt 0) {
        $actualBytes = (Get-Item -LiteralPath $OutFile).Length
        if ($actualBytes -ne $ExpectedBytes) {
            throw "Downloaded size mismatch for $OutFile ($actualBytes / $ExpectedBytes bytes). Re-run the script to resume."
        }
    }
}

function Expand-Zip {
    param(
        [string]$Archive,
        [string]$Destination
    )

    if ($NoExtract) {
        return
    }

    Ensure-Directory $Destination

    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if ($tar) {
        Write-Host "Extracting: $Archive"
        & $tar.Source -xf $Archive -C $Destination
        if ($LASTEXITCODE -ne 0) {
            throw "Extraction failed: $Archive"
        }
        return
    }

    Write-Host "Extracting with Expand-Archive: $Archive"
    Expand-Archive -Force -Path $Archive -DestinationPath $Destination
}

function Ensure-GdownPython {
    $VenvDir = Join-Path $RepoRoot ".download_venv"
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"

    if (-not (Test-Path $VenvPython)) {
        $python = Get-Command python.exe -ErrorAction SilentlyContinue
        if (-not $python) {
            $python = Get-Command python -ErrorAction SilentlyContinue
        }
        if (-not $python) {
            throw "Python not found. SICE Google Drive download needs Python + gdown."
        }

        Write-Host "Creating local download venv: $VenvDir"
        & $python.Source -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create local download venv."
        }
    }

    & $VenvPython -c "import gdown" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing gdown into local download venv"
        & $VenvPython -m pip install -U pip gdown
        if ($LASTEXITCODE -ne 0) {
            throw "Could not install gdown."
        }
    }

    return $VenvPython
}

function Download-GoogleDrive {
    param(
        [string]$FileId,
        [string]$OutFile
    )

    Ensure-Directory (Split-Path -Parent $OutFile)

    if ((Test-Path $OutFile) -and -not $Force) {
        Write-Host "Skip existing: $OutFile"
        return
    }

    $python = Ensure-GdownPython
    $url = "https://drive.google.com/uc?id=$FileId"

    Write-Host "Downloading Google Drive file: $FileId"
    & $python -m gdown $url -O $OutFile
    if ($LASTEXITCODE -ne 0) {
        throw "Google Drive download failed: $FileId"
    }
}

function Try-Expand-Rar {
    param(
        [string]$Archive,
        [string]$Destination
    )

    if ($NoExtract) {
        return
    }

    $sevenZip = Get-Command 7z.exe -ErrorAction SilentlyContinue
    if (-not $sevenZip) {
        Write-Warning "Downloaded $Archive, but 7z.exe was not found. Install 7-Zip to extract .rar files."
        return
    }

    Ensure-Directory $Destination
    Write-Host "Extracting: $Archive"
    & $sevenZip.Source x "-o$Destination" -y $Archive
    if ($LASTEXITCODE -ne 0) {
        throw "Extraction failed: $Archive"
    }
}

function Save-ManualLinks {
    param(
        [string]$DatasetName,
        [string]$Directory,
        [string[]]$Lines
    )

    Ensure-Directory $Directory
    $path = Join-Path $Directory "MANUAL_DOWNLOAD_LINKS.txt"
    $content = @(
        "$DatasetName manual download links",
        "These official sources require Baidu Netdisk or browser-based access.",
        "Download the archives here, then unpack/reorganize them before building manifests.",
        ""
    ) + $Lines

    Set-Content -Encoding UTF8 -Path $path -Value $content
    Write-Warning "$DatasetName is not directly downloadable from a stable public URL. Wrote links to $path"
}

function Download-Kalantari {
    $outDir = Join-Path $RootPath "kalantari"
    Ensure-Directory $outDir

    $testZip = Join-Path $outDir "SIGGRAPH17_HDR_Testset.zip"
    $trainZip = Join-Path $outDir "SIGGRAPH17_HDR_Trainingset.zip"

    Download-Url `
        -Url "https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Testset.zip" `
        -OutFile $testZip `
        -ExpectedBytes 602667980
    Download-Url `
        -Url "https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Trainingset.zip" `
        -OutFile $trainZip `
        -ExpectedBytes 2102116177

    Expand-Zip -Archive $testZip -Destination (Join-Path $outDir "test")
    Expand-Zip -Archive $trainZip -Destination (Join-Path $outDir "train")
}

function Download-Sice {
    $outDir = Join-Path $RootPath "sice"
    Ensure-Directory $outDir

    $part1 = Join-Path $outDir "Dataset_Part1.rar"
    $part2 = Join-Path $outDir "Dataset_Part2.rar"

    Download-GoogleDrive -FileId "1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN" -OutFile $part1
    Download-GoogleDrive -FileId "16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN" -OutFile $part2

    Try-Expand-Rar -Archive $part1 -Destination (Join-Path $outDir "part1")
    Try-Expand-Rar -Archive $part2 -Destination (Join-Path $outDir "part2")
}

function Write-RealHdrvLinks {
    $outDir = Join-Path $RootPath "real_hdrv"
    Save-ManualLinks -DatasetName "Real-HDRV" -Directory $outDir -Lines @(
        "Real-HDRV original RAW: https://pan.baidu.com  code: ab13",
        "Real-HDRV-v1 sRGB HDR video reconstruction: https://pan.baidu.com/s/1aVPxg_KtRjRuzDahDx5ZlQ?pwd=accp  code: accp",
        "Real-HDRV-v2 sRGB HDR deghosting: https://pan.baidu.com/s/1SSjVJLzbv7YyUOF4xdrV1w?pwd=yc5a  code: yc5a"
    )
}

function Write-DeepHdrVideoLinks {
    $outDir = Join-Path $RootPath "deephdrvideo"
    Save-ManualLinks -DatasetName "DeepHDRVideo" -Directory $outDir -Lines @(
        "DeepHDRVideo official BaiduYun share: https://pan.baidu.com/s/19SkOFmOdlQTujuazMlUq2Q?pwd=xwmq  code: xwmq",
        "Inside the share, the project references Real_Dataset/Dynamic/ dynamic_RGB_data_2exp_release.tgz and dynamic_RGB_data_3exp_release.tgz for dynamic scenes with GT HDR.",
        "It also references Synthetic_Dataset/HDR_Synthetic_Test_Dataset.tgz for synthetic testing."
    )
}

$selected = Resolve-DatasetSelection $Dataset

if ($ListOnly) {
    Write-Host "Will handle datasets: $($selected -join ', ')"
    Write-Host "Root: $RootPath"
    exit 0
}

Ensure-Directory $RootPath

foreach ($name in $selected) {
    switch ($name) {
        "kalantari" { Download-Kalantari }
        "sice" { Download-Sice }
        "real-hdrv" { Write-RealHdrvLinks }
        "deephdrvideo" { Write-DeepHdrVideoLinks }
    }
}

Write-Host "Done."
