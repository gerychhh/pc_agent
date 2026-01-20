$p = Get-Process notepad -ErrorAction SilentlyContinue
if ($p) { exit 0 } else { Write-Output "NOT RUNNING"; exit 1 }
