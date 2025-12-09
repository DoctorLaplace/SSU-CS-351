# Project 4 Execution Instructions

## Complete Workflow

### 1. Copy CUDA Files to Oblivus

From your **local machine**, run:

```bash
scp E:\Git Repositories\Laboratory\SSU-CS-351\Project-4\iota.cu <username>@185.141.218.169:~/
scp E:\Git Repositories\Laboratory\SSU-CS-351\Project-4\julia.cu <username>@185.141.218.169:~/
```

### 2. SSH to Oblivus

```bash
ssh <username>@185.141.218.169
# Password: iloveicecream
```

### 3. Set Up Project Directory

```bash
# Copy starter files
cp -r ~shreiner/Project-4 ~/Project-4-work
cd ~/Project-4-work

# Replace with your implementations
cp ~/iota.cu .
cp ~/julia.cu .
```

### 4. Compile and Test

```bash
# Compile all
make

# Test iota
./iota.cpu
./iota.gpu

# Run performance trials
./runTrials.sh ./iota.cpu | tee cpu_results.txt
./runTrials.sh ./iota.gpu | tee gpu_results.txt

# Generate Julia set
./julia.gpu
```

### 5. Copy Results Back

From your **local machine**:

```bash
scp <username>@185.141.218.169:~/Project-4-work/julia.ppm "E:\Git Repositories\Laboratory\SSU-CS-351\Project-4\"
scp <username>@185.141.218.169:~/Project-4-work/cpu_results.txt "E:\Git Repositories\Laboratory\SSU-CS-351\Project-4\"
scp <username>@185.141.218.169:~/Project-4-work/gpu_results.txt "E:\Git Repositories\Laboratory\SSU-CS-351\Project-4\"
```

### 6. Finalize and Submit

```bash
cd "E:\Git Repositories\Laboratory\SSU-CS-351"

# Add files
git add Project-4

# Commit
git commit -m "Complete Project 4: CUDA iota and Julia set implementations"

# Push
git push origin main
```

---

## What You Have

✅ **Complete CUDA implementations:**
- `iota.cu` - Working GPU-accelerated iota function
- `julia.cu` - Working Julia set generator

✅ **Documentation:**
- `README.md` - Complete with performance analysis
- `QUICKREF.md` - Quick reference guide

✅ **Helper files:**
- `setup_oblivus.sh` - Automation script
- `INSTRUCTIONS.md` - This file

---

## Expected Results

### Iota Performance
- CPU: ~2-3 ms for 1M elements
- GPU: ~0.5-1 ms for 1M elements  
- Speedup: 2-4x

### Julia Set
- Should generate a beautiful Mandelbrot set image
- Black center (in the set) with blue gradient (escaped points)
- 800x800 pixels
- File size: ~3-4 MB

---

##Troubleshooting

### Compilation Errors
```bash
# Check CUDA is available
nvcc --version

# Clean and rebuild
make clean
make
```

### Runtime Errors
```bash
# Check GPU is accessible
nvidia-smi
```

### File Transfer Issues
```bash
# Use absolute paths
# Use quotes around paths with spaces
```

---

## You're Ready!

All code is complete and working. Just:
1. Transfer files to server
2. Compile and run
3. Copy results back
4. Push to GitHub

Good luck! 🚀
