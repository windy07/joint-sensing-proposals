param(
    [ValidateSet('check','run')][string]$Stage='run',
    [string]$Project='D:\LXR\radar_stage6_signal_fix',
    [string]$Output='D:\LXR\SERVER_STAGE6_SIGNAL_FIX_12800',
    [string]$RunPython='D:\LXR\conda_envs\radar-stage5\python.exe'
)
$ErrorActionPreference='Stop'
if (-not (Test-Path -LiteralPath $RunPython -PathType Leaf)) { throw "Python not found: $RunPython" }
if (-not (Test-Path -LiteralPath (Join-Path $Output 'RUN_BINDING.json') -PathType Leaf)) { throw "Existing run not found: $Output" }
$RadarArguments=@('-B','-X','utf8',(Join-Path $PSScriptRoot 'epoch10_runner.py'),'--stage',$Stage,'--project',$Project,'--output',$Output)
if ($Stage -eq 'run') { $RadarArguments+='--execute' }
& $RunPython @RadarArguments
if ($LASTEXITCODE -ne 0) { throw "Epoch-10 $Stage failed. Read the complete error above and $Output\EPOCH10_RUN\logs or attempts. Existing checkpoints are retained." }
