[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$submodulePath = '.agents/sources/agent-skills'

git -C $repoRoot submodule update --init -- $submodulePath
if ($LASTEXITCODE -ne 0) { throw 'Failed to initialize agent-skills.' }

& (Join-Path $repoRoot "$submodulePath/scripts/link-skills.ps1") `
    -TargetDirectory (Join-Path $repoRoot '.agents/skills')
