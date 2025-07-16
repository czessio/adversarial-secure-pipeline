#!/usr/bin/env python
"""
Cleanup script to remove all pipeline artifacts and start fresh
"""

import os
import shutil
from pathlib import Path
import argparse


def cleanup_pipeline(dry_run=False, verbose=True):
    """
    Clean up all pipeline-generated files and directories.
    
    Args:
        dry_run: If True, only show what would be deleted without actually deleting
        verbose: If True, print detailed information
    """
    # Directories to clean
    dirs_to_clean = [
        'experiments',
        'results',
        'models/checkpoints',
        'data/raw',  # Downloaded datasets
        'data/processed',
        '__pycache__',
        '.pytest_cache',
    ]
    
    # File patterns to clean
    file_patterns = [
        '**/*.pyc',
        '**/*.pyo',
        '**/__pycache__',
        '**/*.log',
        '**/.ipynb_checkpoints',
        '**/events.out.tfevents.*',  # TensorBoard files
    ]
    
    # Specific files to clean
    files_to_clean = [
        '.coverage',
        'coverage.xml',
        '*.egg-info',
    ]
    
    total_removed = 0
    total_size = 0
    
    print("🧹 Adversarial Robustness Pipeline Cleanup")
    print("=" * 50)
    
    if dry_run:
        print("🔍 DRY RUN MODE - Nothing will be deleted")
        print("=" * 50)
    
    # Clean directories
    print("\n📁 Cleaning directories:")
    for dir_path in dirs_to_clean:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                size = get_dir_size(path)
                total_size += size
                
                if verbose:
                    print(f"  • {dir_path} ({format_size(size)})")
                
                if not dry_run:
                    shutil.rmtree(path, ignore_errors=True)
                    # Recreate important directories
                    if dir_path in ['experiments', 'results', 'models/checkpoints', 'data']:
                        path.mkdir(parents=True, exist_ok=True)
                        # Add .gitkeep files
                        (path / '.gitkeep').touch()
                
                total_removed += 1
    
    # Clean file patterns
    print("\n📄 Cleaning files by pattern:")
    for pattern in file_patterns:
        for file_path in Path('.').glob(pattern):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                
                if verbose:
                    print(f"  • {file_path} ({format_size(size)})")
                
                if not dry_run:
                    file_path.unlink()
                
                total_removed += 1
            elif file_path.is_dir() and '__pycache__' in str(file_path):
                size = get_dir_size(file_path)
                total_size += size
                
                if verbose:
                    print(f"  • {file_path}/ ({format_size(size)})")
                
                if not dry_run:
                    shutil.rmtree(file_path, ignore_errors=True)
                
                total_removed += 1
    
    # Clean specific files
    print("\n🗑️  Cleaning specific files:")
    for file_name in files_to_clean:
        for file_path in Path('.').glob(file_name):
            if file_path.exists():
                size = file_path.stat().st_size if file_path.is_file() else get_dir_size(file_path)
                total_size += size
                
                if verbose:
                    print(f"  • {file_path} ({format_size(size)})")
                
                if not dry_run:
                    if file_path.is_file():
                        file_path.unlink()
                    else:
                        shutil.rmtree(file_path)
                
                total_removed += 1
    
    # Summary
    print("\n" + "=" * 50)
    if dry_run:
        print(f"🔍 Would remove: {total_removed} items ({format_size(total_size)})")
        print("💡 Run without --dry-run to actually clean")
    else:
        print(f"✅ Removed: {total_removed} items ({format_size(total_size)})")
        print("🚀 Pipeline is now clean and ready to run!")
    
    print("\nNext step: Run the pipeline with:")
    print("python scripts/run_pipeline.py --full-pipeline --experiment-name fresh_run --dev-mode")


def get_dir_size(path):
    """Calculate total size of a directory."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except (OSError, IOError):
        pass
    return total


def format_size(size):
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up adversarial robustness pipeline artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup.py              # Clean everything
  python cleanup.py --dry-run    # See what would be cleaned
  python cleanup.py --quiet      # Clean without detailed output
        """
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    cleanup_pipeline(dry_run=args.dry_run, verbose=not args.quiet)