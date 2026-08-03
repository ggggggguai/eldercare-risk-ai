param(
    [int]$Port = 8899
)

$ErrorActionPreference = "Stop"
$distro = "Ubuntu-24.04"
if ($Port -ne 8899) {
    throw "The installed systemd unit is configured for port 8899."
}
& wsl.exe -d $distro -u root -- systemctl stop fun-asr-nano.service
if ($LASTEXITCODE -ne 0) {
    throw "Failed to stop Fun-ASR-Nano in WSL."
}
& wsl.exe -d $distro -u root -- bash -lc "pkill -TERM -f '^sleep infinity$' || true"
Write-Output "Fun-ASR-Nano on port $Port is stopped."
