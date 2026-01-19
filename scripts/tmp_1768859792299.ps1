$process = Get-Process -Name "YandexMusic" -ErrorAction SilentlyContinue

if ($process) {
    $process | Stop-Process
} else {
    Start-Process "C:\Program Files\Yandex Music\YandexMusic.exe"
}