& "D:\10 Work\Programming\20 VENVs\CLI-Bench\Scripts\activate.ps1"

$OGLocation = Get-Location

Set-Location -Path "cli_bench/common"
.\generate_docs.ps1

# Change back to the original location
Set-Location $OGLocation