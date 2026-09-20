param([string]$Output='D:\LXR\SERVER_STAGE6_SIGNAL_FIX_12800',[int]$Seconds=10,[switch]$Once,[switch]$NoGpu)
function Read-InputJson([string]$Path) {
    $RadarStream=$null; $RadarReader=$null
    try {
        $RadarShare=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
        $RadarStream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$RadarShare)
        $RadarReader=[IO.StreamReader]::new($RadarStream,[Text.Encoding]::UTF8)
        return ($RadarReader.ReadToEnd() | ConvertFrom-Json)
    } catch { return $null } finally {
        if ($null -ne $RadarReader) { $RadarReader.Dispose() } elseif ($null -ne $RadarStream) { $RadarStream.Dispose() }
    }
}
do {
    if (-not $Once) { Clear-Host }
    Write-Host "Stage6 input  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Output"
    $RadarNow=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0
    Get-ChildItem -Path (Join-Path $Output 'models\*\*\TRAIN_RECEIPT.json') -File -ErrorAction SilentlyContinue | ForEach-Object {
        $RadarTrain=Read-InputJson $_.FullName
        $RadarInput=Read-InputJson (Join-Path $_.Directory.FullName 'INPUT_PIPELINE_STATUS.json')
        if ($null -ne $RadarTrain) {
            $RadarElapsed=[double]$RadarTrain.elapsed_s
            $RadarPhase=$RadarTrain.phase
            if ($null -ne $RadarInput -and $RadarTrain.status -eq 'RUNNING' -and ($RadarNow-[double]$RadarInput.updated_ns/1e9) -lt 20 -and $RadarInput.consumer_waiting) { $RadarPhase='WAIT_INPUT' }
            [pscustomobject]@{
                Model=$_.Directory.Parent.Name; Seed=$_.Directory.Name; Phase=$RadarPhase
                Updates=$RadarTrain.optimizer_updates; Target=$RadarTrain.target_updates
                RecentCtx_s=[math]::Round([double]$RadarTrain.recent_contexts_per_s,1)
                AverageCtx_s=[math]::Round([double]$RadarTrain.contexts_per_s,1)
                WaitPct=$(if ($RadarElapsed -gt 0) { [math]::Round(100*[double]$RadarTrain.timing.data_wait_s/$RadarElapsed,1) } else { 0 })
                PeakGiB=[math]::Round([double]$RadarTrain.gpu_peak_allocated_bytes/1GB,2)
                ReceiptAge_s=[math]::Round($RadarNow-[double]$RadarTrain.updated_ns/1e9,1)
                LoadedWorlds=$RadarInput.worlds_loaded; Status=$RadarTrain.status
            }
        }
    } | Format-Table -AutoSize
    if (-not $NoGpu) { & nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader }
    if (-not $Once) { Start-Sleep -Seconds ([math]::Max(2,$Seconds)) }
} while (-not $Once)
