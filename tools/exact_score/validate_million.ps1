Add-Type -Path "$PSScriptRoot/MillionScoreValidation.cs"
[MillionScoreValidation]::Run((Join-Path $PSScriptRoot 'comparison.tsv'), (Join-Path $PSScriptRoot 'million_results.tsv'),1000000,20261104)
