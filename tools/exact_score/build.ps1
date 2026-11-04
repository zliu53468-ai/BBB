Add-Type -Path "$PSScriptRoot/ExactScoreBuild.cs"
[ExactScoreBuild]::Run((Join-Path $PSScriptRoot 'exact_scores.tsv'))
