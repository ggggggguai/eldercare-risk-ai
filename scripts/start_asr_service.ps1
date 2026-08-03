param(
    [string]$BindHost = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8091
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    conda run -n eldercare-asr python -m uvicorn `
        elderly_monitoring.modules.asr.api:app `
        --host $BindHost `
        --port $Port `
        --workers 1
}
finally {
    Pop-Location
}
