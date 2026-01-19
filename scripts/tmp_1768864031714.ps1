$desktop = [Environment]::GetFolderPath('Desktop')
$files = Get-ChildItem -Path $desktop
foreach ($file in $files) {
    Write-Host $file.Name
}