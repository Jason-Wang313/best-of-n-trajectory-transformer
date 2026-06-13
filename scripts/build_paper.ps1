$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PaperDir = Join-Path $Root "paper"
$FinalDir = Join-Path $PaperDir "final"
$DownloadsPdf = Join-Path $HOME "Downloads\iclr_submission_trajectory_token_sieve.pdf"
$DesktopDir = Join-Path $HOME "OneDrive\Desktop"
$DesktopPdf = Join-Path $DesktopDir "best of n trajectory transformer-v2.pdf"

New-Item -ItemType Directory -Force $FinalDir | Out-Null
New-Item -ItemType Directory -Force $DesktopDir | Out-Null

Push-Location $PaperDir
try {
    Remove-Item -Force "main.aux", "main.bbl", "main.blg", "main.log", "main.out" -ErrorAction SilentlyContinue
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    Copy-Item -LiteralPath "main.pdf" -Destination (Join-Path $FinalDir "iclr_submission.pdf") -Force
    Copy-Item -LiteralPath "main.pdf" -Destination $DownloadsPdf -Force
    Copy-Item -LiteralPath "main.pdf" -Destination $DesktopPdf -Force
}
finally {
    Pop-Location
}

Write-Host "Saved paper\final\iclr_submission.pdf"
Write-Host "Saved $DownloadsPdf"
Write-Host "Saved $DesktopPdf"
