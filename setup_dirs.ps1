$dirs = @(
    "backend/api", "backend/core", "backend/models", "backend/pipelines",
    "backend/graph", "backend/reasoning", "backend/evaluation",
    "frontend", "worker", "data/raw", "data/processed",
    "config", "docs", "docker", "tests"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
$pkgs = @(
    "backend", "backend/api", "backend/core", "backend/models",
    "backend/pipelines", "backend/graph", "backend/reasoning",
    "backend/evaluation", "worker", "tests"
)
foreach ($p in $pkgs) {
    New-Item -ItemType File -Force -Path "$p/__init__.py" | Out-Null
}
Write-Host "Directory structure created successfully."
