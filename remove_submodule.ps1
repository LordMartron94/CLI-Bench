param(
    [string]$submodulePath
)

# Check if the submodule path is provided
if (-not $submodulePath) {
    Write-Error "Please provide the path to the submodule using the -submodulePath parameter."
    exit 1
}

# 1. Remove the submodule entry from .gitmodules
$gitmodulesPath = Join-Path -Path (Get-Location) -ChildPath ".gitmodules"
if (Test-Path $gitmodulesPath) {
    (Get-Content $gitmodulesPath) | Where-Object { $_ -notmatch "\[submodule ""$submodulePath""\]" } | Set-Content $gitmodulesPath
}

# 2. Remove the submodule entry from .git/config
$gitConfigPath = Join-Path -Path (Get-Location) -ChildPath ".git/config"
if (Test-Path $gitConfigPath) {
    (Get-Content $gitConfigPath) | Where-Object { $_ -notmatch "\[submodule ""$submodulePath""\]" } | Set-Content $gitConfigPath
}

# 3. Remove the submodule from the index
git rm --cached $submodulePath

# 4. Commit the changes
git commit -m "Removed submodule $submodulePath"

# 5. Delete the submodule's files
Remove-Item -Path $submodulePath -Recurse -Force

# 6. Remove the submodule's .git directory
$submoduleGitDir = Join-Path -Path (Get-Location) -ChildPath ".git/modules/$submodulePath"
if (Test-Path $submoduleGitDir) {
    Remove-Item -Path $submoduleGitDir -Recurse -Force
}

Write-Host "Submodule '$submodulePath' removed successfully."