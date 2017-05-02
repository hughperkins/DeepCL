rem draft...

rmdir /s /q "%TEMP%\testmanifest"

mkdir "%TEMP%\testmanifest"
echo "%TEMP%\testmanifest"

call dist\bin\activate.bat

set MNISTDIR=c:\mnist
set MNISTFILE=%MNISTDIR%\train-images-idx3-ubyte

mnist-to-jpegs %MNISTFILE% "%TEMP%\testmanifest" 1280

deepcl_train datadir="%TEMP%\testmanifest" trainfile=manifest.txt validatefile=manifest.txt numtrain=1280 numtest=1280 learningrate=0.002 numepochs=3

rem sed -i -e "s%.*testmanifest/%%g" "%TEMP%\testmanifest"/manifest.txt
rem deepcl_train datadir="%TEMP%\testmanifest" trainfile=manifest.txt validatefile=manifest.txt numtrain=1280 numtest=1280 learningrate=0.002 numepochs=3

rem head -n 1 "%TEMP%\testmanifest"/manifest.txt > "%TEMP%\testmanifest"/test.txt
rem tail -n +2 "%TEMP%\testmanifest"/manifest.txt | awk '{print $1}' >> "%TEMP%\testmanifest"/test.txt
rem deepcl_predict writelabels=1 inputfile="%TEMP%\testmanifest"/test.txt outputfile="%TEMP%\testmanifest"/out.txt

rem head -n 10 "%TEMP%\testmanifest"/test.txt > "%TEMP%\testmanifest"/test_short.txt
rem sed -i -e 's/N=1280/N=9/' "%TEMP%\testmanifest"/test_short.txt
rem deepcl_predict writelabels=1 inputfile="%TEMP%\testmanifest"/test_short.txt outputfile="%TEMP%\testmanifest"/out_short.txt

rem sed -i -e 's/N=9/N=1/' "%TEMP%\testmanifest"/test_short.txt
rem deepcl_predict writelabels=1 inputfile="%TEMP%\testmanifest"/test_short.txt outputfile="%TEMP%\testmanifest"/out_short.txt
