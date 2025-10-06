#!/usr/bin/env python3
"""
Memory Log Analyzer
Analyzes memory log files to find maximum GPU and CPU memory usage.
"""

import re
import sys
import os
from typing import Tuple, List, Dict, Optional
import argparse


class MemoryPhase:
    """Represents a phase of memory usage (compilation or inference)."""
    def __init__(self, name: str):
        self.name = name
        self.data_points = []
        self.max_gpu = 0.0
        self.max_cpu = 0.0
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None


def parse_memory_log(file_path: str) -> Tuple[float, float, List[Dict], Dict[str, MemoryPhase]]:
    """
    Parse memory log file and extract GPU and CPU memory usage data.
    
    Args:
        file_path: Path to the memory log file
        
    Returns:
        Tuple containing (max_gpu_memory, max_cpu_memory, all_data_points, phases_dict)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Memory log file not found: {file_path}")
    
    max_gpu_memory = 0.0
    max_cpu_memory = 0.0
    data_points = []
    
    # Track phases
    phases = {
        'compilation': MemoryPhase('Compilation'),
        'inference': MemoryPhase('Inference'),
        'overall': MemoryPhase('Overall')
    }
    
    current_phase = 'compilation'  # Start with compilation phase
    
    # Regular expressions to match memory usage lines
    gpu_pattern = re.compile(r'GPU Memory Usage: ([\d.]+) MB')
    cpu_pattern = re.compile(r'CPU Memory Usage: ([\d.]+) MB')
    timestamp_pattern = re.compile(r'Timestamp: ([\d:.]+)')
    compilation_finished_pattern = re.compile(r'Compilation finished\.')
    
    current_timestamp = None
    current_gpu_memory = 0.0
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Check for compilation finished marker
            if compilation_finished_pattern.search(line):
                current_phase = 'inference'
                continue
            
            # Check for timestamp
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                current_timestamp = timestamp_match.group(1)
                if phases[current_phase].start_time is None:
                    phases[current_phase].start_time = current_timestamp
                phases[current_phase].end_time = current_timestamp
                continue
            
            # Check for GPU memory usage
            gpu_match = gpu_pattern.search(line)
            if gpu_match:
                current_gpu_memory = float(gpu_match.group(1))
                max_gpu_memory = max(max_gpu_memory, current_gpu_memory)
                phases[current_phase].max_gpu = max(phases[current_phase].max_gpu, current_gpu_memory)
                continue
            
            # Check for CPU memory usage
            cpu_match = cpu_pattern.search(line)
            if cpu_match:
                cpu_memory = float(cpu_match.group(1))
                max_cpu_memory = max(max_cpu_memory, cpu_memory)
                phases[current_phase].max_cpu = max(phases[current_phase].max_cpu, cpu_memory)
                
                # Store data point if we have all information
                if current_timestamp:
                    data_point = {
                        'timestamp': current_timestamp,
                        'gpu_memory': current_gpu_memory,
                        'cpu_memory': cpu_memory,
                        'phase': current_phase
                    }
                    data_points.append(data_point)
                    phases[current_phase].data_points.append(data_point)
                    phases['overall'].data_points.append(data_point)
                continue
    
    # Set overall phase stats
    phases['overall'].max_gpu = max_gpu_memory
    phases['overall'].max_cpu = max_cpu_memory
    if data_points:
        phases['overall'].start_time = data_points[0]['timestamp']
        phases['overall'].end_time = data_points[-1]['timestamp']
    
    return max_gpu_memory, max_cpu_memory, data_points, phases


def format_memory_size(memory_mb: float) -> str:
    """Convert memory size in MB to a human-readable format."""
    if memory_mb >= 1024:
        return f"{memory_mb / 1024:.2f} GB ({memory_mb:.2f} MB)"
    else:
        return f"{memory_mb:.2f} MB"


def extract_model_name(file_path: str) -> str:
    """Extract model name from memory log filename."""
    filename = os.path.basename(file_path)
    # Expected format: memory_log_[model_name]_[date].txt
    if filename.startswith("memory_log_") and filename.endswith(".txt"):
        # Remove prefix and suffix
        name_part = filename[11:-4]  # Remove "memory_log_" and ".txt"
        # The date part should be in format YYYYMMDD_HHMMSS at the end
        # Look for pattern like _20250930_154414 at the end
        import re
        date_pattern = r'_\d{8}_\d{6}$'
        match = re.search(date_pattern, name_part)
        if match:
            # Remove the date part
            model_name = name_part[:match.start()]
            return model_name
        else:
            # If no date pattern found, assume last underscore separated part is date
            parts = name_part.split('_')
            if len(parts) > 1:
                model_name = '_'.join(parts[:-1])
                return model_name
    return "Unknown Model"


def analyze_memory_usage(file_path: str, verbose: bool = False) -> None:
    """
    Analyze memory usage from log file and display results.
    
    Args:
        file_path: Path to the memory log file
        verbose: Whether to show detailed statistics
    """
    try:
        max_gpu, max_cpu, data_points, phases = parse_memory_log(file_path)
        
        # Calculate idle memory (minimum usage) to subtract from all measurements
        gpu_values = [dp['gpu_memory'] for dp in data_points]
        cpu_values = [dp['cpu_memory'] for dp in data_points]
        
        idle_gpu = min(gpu_values) if gpu_values else 0
        idle_cpu = min(cpu_values) if cpu_values else 0
        
        # Calculate net memory usage (subtracting idle)
        net_max_gpu = max_gpu - idle_gpu
        net_max_cpu = max_cpu - idle_cpu
        
        # Extract model name from filename
        model_name = extract_model_name(file_path)
        
        print(f"Memory Usage Analysis for: {model_name}")
        print(f"Log file: {os.path.basename(file_path)}")
        print("=" * 60)
        print(f"Idle Memory (baseline): GPU {format_memory_size(idle_gpu)}, CPU {format_memory_size(idle_cpu)}")
        print()
        print(f"Net Maximum GPU Memory Usage: {format_memory_size(net_max_gpu)}")
        print(f"Net Maximum CPU Memory Usage: {format_memory_size(net_max_cpu)}")
        print(f"Total data points analyzed: {len(data_points)}")
        
        # Phase-specific analysis with net memory (always shown)
        print(f"\nPhase-specific Analysis (Net Memory Usage):")
        print("-" * 45)
        
        for phase_name, phase in phases.items():
            if phase_name == 'overall' or not phase.data_points:
                continue
                
            net_phase_gpu = phase.max_gpu - idle_gpu
            net_phase_cpu = phase.max_cpu - idle_cpu
            
            print(f"\n{phase.name} Phase:")
            print(f"  Duration: {phase.start_time} - {phase.end_time}")
            print(f"  Data points: {len(phase.data_points)}")
            print(f"  Net Peak GPU: {format_memory_size(net_phase_gpu)}")
            print(f"  Net Peak CPU: {format_memory_size(net_phase_cpu)}")
        
        if verbose:
            # Verbose mode - show detailed statistics
            _show_detailed_statistics(data_points, gpu_values, cpu_values, 
                                   idle_gpu, idle_cpu, 
                                   net_max_gpu, net_max_cpu)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing memory log: {e}")
        sys.exit(1)


def _show_detailed_statistics(data_points, gpu_values, cpu_values,
                            idle_gpu, idle_cpu,
                            net_max_gpu, net_max_cpu):
    """Show detailed statistics in verbose mode."""
    print("\nDetailed Statistics (Net Memory Usage):")
    print("-" * 45)
    
    # Calculate net values for detailed stats
    net_gpu_values = [gpu - idle_gpu for gpu in gpu_values]
    net_cpu_values = [cpu - idle_cpu for cpu in cpu_values]
    
    if net_gpu_values:
        avg_net_gpu = sum(net_gpu_values) / len(net_gpu_values)
        min_net_gpu = min(net_gpu_values)
        print(f"Net GPU Memory - Min: {format_memory_size(min_net_gpu)}, "
              f"Avg: {format_memory_size(avg_net_gpu)}, "
              f"Max: {format_memory_size(net_max_gpu)}")
    
    if net_cpu_values:
        avg_net_cpu = sum(net_cpu_values) / len(net_cpu_values)
        min_net_cpu = min(net_cpu_values)
        print(f"Net CPU Memory - Min: {format_memory_size(min_net_cpu)}, "
              f"Avg: {format_memory_size(avg_net_cpu)}, "
              f"Max: {format_memory_size(net_max_cpu)}")
    
    # Find peak usage timestamps with net memory
    peak_gpu_point = max(data_points, key=lambda x: x.get('gpu_memory', 0))
    peak_cpu_point = max(data_points, key=lambda x: x['cpu_memory'])
    
    net_peak_gpu_usage = peak_gpu_point['gpu_memory'] - idle_gpu
    net_peak_cpu_usage = peak_cpu_point['cpu_memory'] - idle_cpu
    
    print(f"\nPeak Usage Timestamps (Net Memory):")
    print(f"Peak Net GPU usage at: {peak_gpu_point['timestamp']} "
          f"({format_memory_size(net_peak_gpu_usage)}) - {peak_gpu_point['phase'].title()}")
    print(f"Peak Net CPU usage at: {peak_cpu_point['timestamp']} "
          f"({format_memory_size(net_peak_cpu_usage)}) - {peak_cpu_point['phase'].title()}")


def find_memory_logs(directory: str = ".") -> List[str]:
    """Find all memory log files in the specified directory."""
    memory_log_files = []
    for file in os.listdir(directory):
        if file.startswith("memory_log") and file.endswith(".txt"):
            memory_log_files.append(os.path.join(directory, file))
    return sorted(memory_log_files)


def main():
    parser = argparse.ArgumentParser(description="Analyze memory usage from log files")
    parser.add_argument("file", nargs="?", help="Memory log file to analyze")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Show detailed statistics (averages, timestamps, etc.)")
    parser.add_argument("-a", "--all", action="store_true",
                       help="Analyze all memory log files in current directory")
    
    args = parser.parse_args()
    
    # Default is now quiet mode (basic + phase analysis), verbose adds detailed stats
    verbose = args.verbose
    
    if args.all:
        # Analyze all memory log files in current directory
        log_files = find_memory_logs()
        if not log_files:
            print("No memory log files found in current directory.")
            return
        
        print(f"Found {len(log_files)} memory log file(s):")
        print()
        
        for log_file in log_files:
            analyze_memory_usage(log_file, verbose)
            print()
    
    elif args.file:
        # Analyze specific file
        analyze_memory_usage(args.file, verbose)
    
    else:
        # Try to find and analyze the most recent memory log file
        log_files = find_memory_logs()
        if log_files:
            print("No file specified. Analyzing most recent memory log file:")
            analyze_memory_usage(log_files[-1], verbose)
        else:
            print("No memory log file specified and none found in current directory.")
            print("Usage: python memory_analyzer.py <memory_log_file>")
            parser.print_help()


if __name__ == "__main__":
    main()
