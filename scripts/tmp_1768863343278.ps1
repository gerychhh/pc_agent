$wordPath = "$env:USERPROFILE\Desktop\test.docx"
New-Item -ItemType File -Path $wordPath
Start-Process "C:\Program Files (x86)\Microsoft Office\Office16\WINWORD.EXE" -ArgumentList "/T:$wordPath"