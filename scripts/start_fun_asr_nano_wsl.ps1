param(
    [int]$Port = 8899,
    [int]$StartupTimeoutSeconds = 240
)

$ErrorActionPreference = "Stop"
$distro = "Ubuntu-24.04"
if ($Port -ne 8899) {
    throw "The installed systemd unit is configured for port 8899."
}

$keepAlive = & wsl.exe -d $distro -u root -- bash -lc "pgrep -f '^sleep infinity$' || true"
if (-not $keepAlive) {
    Start-Process -FilePath "wsl.exe" `
        -ArgumentList @("-d", $distro, "-u", "root", "--", "sleep", "infinity") `
        -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

& wsl.exe -d $distro -u root -- systemctl start fun-asr-nano.service
if ($LASTEXITCODE -ne 0) {
    throw "Failed to launch Fun-ASR-Nano in WSL."
}

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
$readyUrl = "http://localhost:$Port/openapi.json"
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri $readyUrl -TimeoutSec 3 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Output "Fun-ASR-Nano is ready: http://localhost:$Port/docs"
            exit 0
        }
    }
    catch {
        Start-Sleep -Seconds 2
    }
}

Write-Output "Fun-ASR-Nano did not become ready. Recent WSL log:"
& wsl.exe -d $distro -u root -- journalctl -u fun-asr-nano.service -n 60 --no-pager
exit 1
