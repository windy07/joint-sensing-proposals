param(
    [ValidateSet('check','develop','train','development','pack')][string]$Stage='check',
    [string]$Project='D:\LXR\radar_stage6_signal_fix',
    [string]$Output='D:\LXR\SERVER_STAGE6_SIGNAL_FIX_12800',
    [ValidateSet(1280,12800)][int]$Scale=12800,
    [string]$RunPython='D:\LXR\conda_envs\radar-stage5\python.exe'
)
$ErrorActionPreference='Stop'
if (-not (Test-Path -LiteralPath $RunPython -PathType Leaf)) { throw "Python not found: $RunPython" }
$RadarInputArgs=@('-B','-X','utf8',(Join-Path $PSScriptRoot 'stage6_input_runner.py'),'--transport-project',$Project)
if ($Stage -eq 'check') { $RadarInputArgs+='--transport-check' }
else {
    $RadarInputArgs+=@($Stage,'--plan',(Join-Path $Project "STAGE6_SERVER\plan_$Scale.json"),'--output',$Output,'--resume')
    if ($Stage -ne 'pack') { $RadarInputArgs+='--execute' }
}
& $RunPython @RadarInputArgs
if ($LASTEXITCODE -ne 0) { throw "Stage6 input adapter $Stage failed. Keep the full error and current output directory." }
