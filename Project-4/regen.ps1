$User = "jhalaweh"
$Server = "185.141.218.169"
$LocalDir = "E:\Git Repositories\Laboratory\SSU-CS-351\Project-4"

Write-Host "--- JULIA SET REGENERATOR ---" -ForegroundColor Cyan

Write-Host "1. Uploading julia.cu..." -ForegroundColor Yellow
scp "$LocalDir\julia.cu" ${User}@${Server}:~/Project-4-work/
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "2. Compiling (Adding CUDA to PATH)..." -ForegroundColor Yellow
ssh ${User}@${Server} "export PATH=/usr/local/cuda/bin:`$PATH; cd ~/Project-4-work; make julia.gpu"
if ($LASTEXITCODE -ne 0) { 
    Write-Host "Compilation Failed!" -ForegroundColor Red
    exit 1 
}

Write-Host "3. Running julia.gpu..." -ForegroundColor Yellow
ssh ${User}@${Server} "cd ~/Project-4-work; ./julia.gpu"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "4. Downloading result..." -ForegroundColor Yellow
scp ${User}@${Server}:~/Project-4-work/julia.ppm "$LocalDir\julia.ppm"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "5. Converting to PNG..." -ForegroundColor Yellow
python -c "from PIL import Image; import os; os.chdir(r'$LocalDir'); img = Image.open('julia.ppm'); img.save('julia.png'); print('Saved julia.png')"

Write-Host "Done!" -ForegroundColor Green