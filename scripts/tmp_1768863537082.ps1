$imagePath = "$env:USERPROFILE\Desktop\any_image.jpg"
if (Test-Path $imagePath) {
    python -c "from PIL import Image; image = Image.open($imagePath); image.show()"
}
else {
    Write-Host "Файл не найден."
}