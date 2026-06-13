$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PaperDir = Join-Path $Root "paper"
$FinalDir = Join-Path $PaperDir "final"
$FinalPdf = Join-Path $FinalDir "best of n trajectory transformer-v3.pdf"

New-Item -ItemType Directory -Force $FinalDir | Out-Null

Push-Location $PaperDir
try {
    Remove-Item -Force "main.aux", "main.bbl", "main.blg", "main.log", "main.out" -ErrorAction SilentlyContinue
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    Copy-Item -LiteralPath "main.pdf" -Destination $FinalPdf -Force
}
finally {
    Pop-Location
}

Write-Host "Saved $FinalPdf"
