$filename = "million.bin"
$results = @()

# Function to measure execution time
function Measure-Run {
    param (
        [string]$command,
        [string]$arguments,
        [int]$threads
    )
    
    $time = Measure-Command {
        $process = Start-Process -FilePath $command -ArgumentList $arguments -NoNewWindow -Wait -PassThru
    }
    
    $obj = [PSCustomObject]@{
        Threads = $threads
        TimeMs  = $time.TotalMilliseconds
    }
    
    Write-Host ("Threads: {0,2} | Time: {1,8:N2} ms" -f $threads, $time.TotalMilliseconds)
    return $obj
}

Write-Host "Running Serial Mean..."
$serialArgs = "-f $filename"
# Note: mean.out doesn't take -f but we'll use it as arg 1 if we modified it, 
# but original mean.cpp takes filename as argv[1].
# Let's check mean.cpp again. It takes argv[1].
$serialArgs = "$filename"
$serialResult = Measure-Run -command "./mean.out" -arguments $serialArgs -threads 1

Write-Host "`nRunning Threaded Mean..."
$threadCounts = 1, 2, 4, 8, 12, 16, 20, 24
foreach ($t in $threadCounts) {
    $threadedArgs = "-f $filename -t $t"
    $results += Measure-Run -command "./threaded.out" -arguments $threadedArgs -threads $t
}

$csvData = @()

Write-Host "`n--- Mean Summary ---"
Write-Host ("Serial Time: {0:N2} ms" -f $serialResult.TimeMs)
foreach ($res in $results) {
    $speedup = $serialResult.TimeMs / $res.TimeMs
    Write-Host ("Threads: {0,2} | Time: {1,8:N2} ms | Speedup: {2:N2}x" -f $res.Threads, $res.TimeMs, $speedup)
    $csvData += [PSCustomObject]@{
        Type    = "Mean"
        Threads = $res.Threads
        TimeMs  = $res.TimeMs
        Speedup = $speedup
    }
}

$sdfResults = @()
$sdfSamples = 100000000

Write-Host "`nRunning Serial SDF..."
$sdfSerialArgs = "-t 1 -n $sdfSamples"
$sdfSerialResult = Measure-Run -command "./sdf.out" -arguments $sdfSerialArgs -threads 1

Write-Host "`nRunning Threaded SDF..."
foreach ($t in $threadCounts) {
    $sdfThreadedArgs = "-t $t -n $sdfSamples"
    $sdfResults += Measure-Run -command "./sdf.out" -arguments $sdfThreadedArgs -threads $t
}

Write-Host "`n--- SDF Summary ---"
Write-Host ("Serial Time: {0:N2} ms" -f $sdfSerialResult.TimeMs)
foreach ($res in $sdfResults) {
    $speedup = $sdfSerialResult.TimeMs / $res.TimeMs
    Write-Host ("Threads: {0,2} | Time: {1,8:N2} ms | Speedup: {2:N2}x" -f $res.Threads, $res.TimeMs, $speedup)
    $csvData += [PSCustomObject]@{
        Type    = "SDF"
        Threads = $res.Threads
        TimeMs  = $res.TimeMs
        Speedup = $speedup
    }
}

$csvData | Export-Csv -Path "results.csv" -NoTypeInformation
