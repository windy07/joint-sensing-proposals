param(
    [string]$Project='D:\LXR\radar_stage6_signal_fix',
    [string]$Output='D:\LXR\SERVER_STAGE6_SIGNAL_FIX_12800'
)
$ErrorActionPreference='Stop'
$RadarProjectFull=[IO.Path]::GetFullPath($Project).TrimEnd('\')
$RadarOutputFull=[IO.Path]::GetFullPath($Output).TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $RadarProjectFull 'stage6_server.py') -PathType Leaf)) { throw 'Project identity file missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $RadarOutputFull 'RUN_BINDING.json') -PathType Leaf)) { throw 'Run binding missing.' }

function Test-RadarCommand([string]$Command) {
    if ([string]::IsNullOrWhiteSpace($Command)) { return $false }
    $RadarIsEntry=$Command -match '(?i)(stage6_input_runner|stage6_server|stage6|epoch10_runner)\.py(?:"|\s)'
    $RadarOutputPattern='(?i)'+[regex]::Escape($RadarOutputFull)+'(?=[\\/"\s]|$)'
    $RadarHasOutput=$Command -match $RadarOutputPattern
    # The worker-spec entry contains the run directory; old worker commands
    # include the project and the output explicitly.
    $RadarProjectPattern='(?i)'+[regex]::Escape($RadarProjectFull)+'(?=[\\/"\s]|$)'
    $RadarHasProject=$Command -match $RadarProjectPattern
    return ($RadarIsEntry -and $RadarHasOutput -and ($RadarHasProject -or $Command -match '(?i)epoch10_runner\.py'))
}

$RadarSnapshot=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { Test-RadarCommand $_.CommandLine })
if ($RadarSnapshot.Count -eq 0) { Write-Host 'No matching radar training processes are running.'; return }
Write-Host 'Stopping these processes for the selected radar run:'
$RadarSnapshot | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | Format-Table -Wrap
# Stop coordinators before their known children, then verify PID and creation
# time again before each termination. Unrelated Python processes are excluded.
$RadarOrdered=@($RadarSnapshot | Sort-Object @{Expression={ if ($_.CommandLine -match '(?i)(--worker-spec|\sworker\s)') { 1 } else { 0 } }})
foreach ($RadarProcess in $RadarOrdered) {
    $RadarCurrent=Get-CimInstance Win32_Process -Filter ('ProcessId={0}' -f $RadarProcess.ProcessId) -ErrorAction SilentlyContinue
    if ($null -eq $RadarCurrent) { continue }
    if ($RadarCurrent.CreationDate -ne $RadarProcess.CreationDate -or -not (Test-RadarCommand $RadarCurrent.CommandLine)) { throw 'Process identity changed; no unrelated process was stopped.' }
    Stop-Process -Id $RadarCurrent.ProcessId -Force -ErrorAction Stop
}
$RadarDeadline=(Get-Date).AddSeconds(30)
do {
    $RadarRemaining=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { Test-RadarCommand $_.CommandLine })
    if ($RadarRemaining.Count -eq 0) { break }
    Start-Sleep -Seconds 1
} while ((Get-Date) -lt $RadarDeadline)
if ($RadarRemaining.Count -gt 0) { throw 'Some selected radar processes are still running. Do not start another coordinator.' }
Write-Host 'Selected radar processes stopped. Data and completed checkpoint files are preserved.'
Write-Host 'Resume may repeat up to 127 optimizer updates since the latest periodic checkpoint.'
