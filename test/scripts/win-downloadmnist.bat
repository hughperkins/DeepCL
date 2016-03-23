powershell Set-ExecutionPolicy unrestricted
powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz')
powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz')
powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz')
powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

"c:\program files\7-Zip\7z.exe" x train-images-idx3-ubyte.gz
"c:\program files\7-Zip\7z.exe" x train-labels-idx1-ubyte.gz
"c:\program files\7-Zip\7z.exe" x t10k-images-idx3-ubyte.gz
"c:\program files\7-Zip\7z.exe" x t10k-labels-idx1-ubyte.gz

