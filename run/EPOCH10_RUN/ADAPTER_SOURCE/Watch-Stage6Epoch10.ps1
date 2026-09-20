param(
    [string]$Output='D:\LXR\SERVER_STAGE6_SIGNAL_FIX_12800',
    [int]$IntervalSeconds=15,
    [switch]$Once
)
$ErrorActionPreference='Stop'
function Read-RadarJson([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $RadarStream=$null; $RadarReader=$null
    try {
        $RadarStream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
        $RadarReader=[IO.StreamReader]::new($RadarStream,[Text.Encoding]::UTF8)
        return ($RadarReader.ReadToEnd() | ConvertFrom-Json)
    } catch { return $null }
    finally {
        if ($null -ne $RadarReader) { $RadarReader.Dispose() }
        elseif ($null -ne $RadarStream) { $RadarStream.Dispose() }
    }
}
do {
    if (-not $Once) { Clear-Host }
    Write-Host ('Stage6 to epoch 10   {0}   {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'),$Output)
    $RadarStatus=Read-RadarJson (Join-Path $Output 'EPOCH10_RUN\STATUS.json')
    if ($null -ne $RadarStatus) {
        $RadarStatus | Select-Object status,complete,total,running,failed,pending | Format-Table -AutoSize
        if ($RadarStatus.error) { Write-Host $RadarStatus.error }
    }
    $RadarRows=@(foreach ($RadarArm in 'integrated_agent','deep_sets','set_transformer') {
        foreach ($RadarSeed in 633431,633432,633433) {
            $RadarDirectory=Join-Path (Join-Path (Join-Path $Output 'models') $RadarArm) ([string]$RadarSeed)
            $RadarTrain=Read-RadarJson (Join-Path $RadarDirectory 'TRAIN_RECEIPT.json')
            $RadarFinal=Read-RadarJson (Join-Path $RadarDirectory 'EPOCH_010_RESULT.json')
            $RadarActive=@($RadarStatus.active | Where-Object { $_.job.arm -eq $RadarArm -and $_.job.model_seed -eq $RadarSeed })
            [pscustomobject]@{
                Model=$RadarArm; Seed=$RadarSeed
                GPU=$(if ($RadarActive.Count -gt 0) { $RadarActive[0].gpu } else { '-' })
                Phase=$(if ($null -ne $RadarFinal) { 'COMPLETE' } elseif ($RadarActive.Count -gt 0) { if ($RadarTrain.phase) { $RadarTrain.phase } else { 'STARTING' } } else { 'PENDING' })
                Updates=$(if ($null -ne $RadarFinal) { $RadarFinal.optimizer_updates } elseif ($null -ne $RadarTrain) { $RadarTrain.optimizer_updates } else { 0 })
                Target=32000
                Epoch=$(if ($null -ne $RadarFinal) { 10 } elseif ($null -ne $RadarTrain.cursor) { $RadarTrain.cursor.epoch } else { 0 })
                RecentCtx_s=$(if ($RadarTrain.recent_contexts_per_s) { [math]::Round($RadarTrain.recent_contexts_per_s,1) } else { 0 })
                ValLoss=$(if ($null -ne $RadarFinal) { [math]::Round($RadarFinal.validation.total,5) } elseif ($null -ne $RadarTrain.validation) { [math]::Round($RadarTrain.validation.total,5) } else { '-' })
            }
        }
    })
    $RadarRows | Format-Table -AutoSize
    & nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    if ($Once -or $RadarStatus.status -in @('COMPLETE','FAILED','INTERRUPTED')) { break }
    Start-Sleep -Seconds ([math]::Max(1,$IntervalSeconds))
} while ($true)
