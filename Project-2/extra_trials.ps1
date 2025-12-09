$samples = 100000000

function Measure-Run {
    param (
        [string]$command,
        [string]$arguments,
        [string]$name
    )
    
    Write-Host "Running $name..."
    $time = Measure-Command {
        $process = Start-Process -FilePath $command -ArgumentList $arguments -NoNewWindow -Wait -PassThru
    }
    
    Write-Host ("Time: {0:N2} ms" -f $time.TotalMilliseconds)
    return $time.TotalMilliseconds
}

$baseline = Measure-Run -command "./sdf.out" -arguments "-t 1 -n $samples" -name "Baseline Serial (sdf.out)"
$fastest = Measure-Run -command "./sdf-fastest.out" -arguments "-n $samples" -name "Fastest Serial (sdf-fastest.out)"
$speedup = Measure-Run -command "./sdf-speedup.out" -arguments "-t 24 -n $samples" -name "Fastest Threaded (sdf-speedup.out)"

Write-Host "`n--- Extra Credit Results ---"
Write-Host ("Baseline Time: {0:N2} ms" -f $baseline)
Write-Host ("Fastest Serial Time: {0:N2} ms | Speedup vs Baseline: {1:N2}x" -f $fastest, ($baseline / $fastest))
Write-Host ("Fastest Threaded Time: {0:N2} ms | Speedup vs Baseline: {1:N2}x | Speedup vs Fastest Serial: {2:N2}x" -f $speedup, ($baseline / $speedup), ($fastest / $speedup))
