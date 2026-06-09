$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\paper")
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
New-Item -ItemType Directory -Force "final" | Out-Null
Copy-Item -LiteralPath "main.pdf" -Destination "final\iclr_submission.pdf" -Force
Copy-Item -LiteralPath "main.pdf" -Destination (Join-Path $HOME "Downloads\iclr_submission_trajectory_transformer.pdf") -Force
Write-Host "Wrote paper\final\iclr_submission.pdf and Downloads copy."
