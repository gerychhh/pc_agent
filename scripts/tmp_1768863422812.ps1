$wordFile = "$env:USERPROFILE\Desktop\2026.docx"
New-Item -Path $wordFile -ItemType File | Out-Null
Add-Type -AssemblyName Microsoft.Office.Interop.Word
$wordApp = New-Object -ComObject Word.Application
$doc = $wordApp.Documents.Add()
$doc.Content.Text = "2026"
$doc.SaveAs($wordFile)
$wordApp.Quit()