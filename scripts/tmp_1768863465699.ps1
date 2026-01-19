$wordPath = "$env:USERPROFILE\Desktop\2026.docx"
if (Test-Path $wordPath) { Remove-Item -Force $wordPath }
New-Item -ItemType File -Path $wordPath | Out-Null
Add-Content -Path $wordPath -Value "2026"