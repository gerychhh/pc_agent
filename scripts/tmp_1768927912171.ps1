Get-ChildItem | Where-Object { $_.Extension -eq ".docx" } | ForEach-Object {
    $filePath = $_.FullName
    Compress-Item -Path $filePath -DestinationFile ($filePath + "_backup.zip") -CompressionLevel OptimizedForSize
}