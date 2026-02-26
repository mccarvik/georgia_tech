import os, argparse, json
import matplotlib.pyplot as plt
import numpy as np
import common


def filter_values_from_stats(key, data):
    """
    Filter a values corresponding to a key from stat dict
    """
    
    tags = []         # name of benchmarks
    values = {}       # config: values for benchmarks

    gpu_configs = list(data.keys())
    benchmarks = list(data[next(iter(data))].keys())
       
    for gpucfg in gpu_configs:
        values[gpucfg] = []

    # Check if we need to skip any benchmarks
    skipped_benchmarks=[]
    for benchmark in benchmarks:
        for gpucfg in gpu_configs:
            benchmark_data = data[gpucfg][benchmark]          
            if key not in benchmark_data.keys():
                print(f'... skipping {benchmark} due to missing key "{key}" in {gpucfg}:{benchmark}')
                if benchmark not in skipped_benchmarks:
                    skipped_benchmarks += [benchmark]

    # Add tag, data
    for benchmark in benchmarks:
        if benchmark in skipped_benchmarks:
            continue # Skip
        tags += [benchmark]
        for gpucfg in gpu_configs:
            values[gpucfg] += [data[gpucfg][benchmark][key]]
    return tags, values


def plot_comparison(benchmarks, data_dict:dict, file, lbl_x='X Axis', lbl_y='Y Axis', title='Untitled'):
    """
    Plot data proided (as dict) using a bar-graph
    """

    # Plot
    width = 0.25  # the width of the bars
    multiplier = 0
    x = np.arange(len(benchmarks))

    fig, ax = plt.subplots(figsize=(30, 20))

    n_series = len(data_dict.keys())

    for series_name in data_dict.keys():    
        series = data_dict[series_name]
        offset = width * multiplier
        bars = ax.bar(x + offset, series, width, capsize=20, label=series_name)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(lbl_y, fontsize=28)
    ax.set_xlabel(lbl_x, fontsize=28)

    ax.set_title(title, fontsize=36, fontweight='bold')
    ax.set_xticks(x+width/n_series) # FIXME: if alignment issue with xticks

    ax.set_xticklabels(benchmarks, rotation=45, fontsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.legend(loc='upper left', fontsize=28)
    
    plt.tight_layout()  # Adjust layout to prevent cropping
    plt.savefig(file)


if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trace-dir', help='Specify trace directory', type=str, default=os.getenv('MACSIM_TRACE_DIR'))
    parser.add_argument('-l', '--log-dir', help='Specify log directory', type=str, default=os.path.join(os.getenv('MACSIM_DIR'), 'log'))
    parser.add_argument('-o', '--output', help='Specify output file', type=str, default='')
    parser.add_argument('json', help='JSON file containing results', type=str)
    parser.add_argument('stat', help='Specify stat to plot', type=str)

    
    args = parser.parse_args()

    trace_dir = args.trace_dir
    log_dir = args.log_dir

    stats_dict={}
    print(f'> Reading JSON {args.json}')
    with open(args.json, 'r') as file:
        stats_dict = json.load(file)

    if args.stat:
        stat=args.stat
        print(f'Plotting: {stat}')

        # Filter values from dict
        benchmark_list, benchmark_stats = filter_values_from_stats(stat, stats_dict)

        ofile=''
        if args.output:
            ofile=args.output
        else:
            ofile=os.path.join(log_dir, f'{stat.lower()}.png')

        print('Writing:', ofile)
        plot_comparison(benchmark_list, benchmark_stats, ofile, title='Warp Scheduling Algorithms Comparison', lbl_x='Benchmarks', lbl_y=stat)

