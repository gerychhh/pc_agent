$q=[uri]::EscapeDataString("два зла")
Start-Process "https://www.youtube.com/results?search_query=$q"
