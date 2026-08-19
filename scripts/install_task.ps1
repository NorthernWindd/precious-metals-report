param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonPath = (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    [string]$TaskName = "PreciousMetalsDailyReport",
    [string]$Time = "08:00"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonPath)) {
    throw "Python path not found: $PythonPath"
}

$actionArguments = '/c "' + 'cd /d "' + $ProjectRoot + '" && "' + $PythonPath + '" -m pmreport --config config.yaml >> logs\daily.log 2>&1"'

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $actionArguments
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Daily precious metals trend report" `
    -Force | Out-Null

Write-Host "Task registered: $TaskName at $Time"
