$p = Get-Process WINWORD -ErrorAction SilentlyContinue
if ($p) { exit 0 } else { exit 1 }
