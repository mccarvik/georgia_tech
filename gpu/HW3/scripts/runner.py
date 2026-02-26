import os, subprocess, time, argparse, json
import common

def run_macsim(benchmarks:dict, gpu_configs:dict, log_dir:str, process_timeout=10*60):
    """
    Run macsim on specified benchmarks and gpu configs, and dump logs
    """

    # Iterate over benchmarks and gpu configs
    all_ok=True
    for gpucfg in gpu_configs.keys():   
        for benchmark in benchmarks.keys():
            print('='*25+'[{:^35s}]'.format(f'Running {benchmark} ({gpucfg})')+'='*25)
            gpu_config = gpu_configs[gpucfg]
            kernel_config = benchmarks[benchmark]
            log_file = os.path.join(log_dir, f'{gpucfg}_{benchmark}_run.log')
            cmd = ['./macsim', '-g', gpu_config, '-t', kernel_config]

            print('GPU cfg    : ', gpu_config)
            print('Kernel cfg : ', kernel_config)
            print('Log file   : ', log_file)
            print('Command    : ', ' '.join(cmd))
            
            with open(log_file, 'w') as log:
                # run macsim as subprocess
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                timeout = time.time() + process_timeout
                i=0
                while time.time() < timeout:
                    if process.poll() is not None:
                        break              
                    print('.', end=('\n' if i%60==59 else ''), flush=True)
                    i+=1
                    time.sleep(1)

                print('\nStatus     :  ', end='')
                if process.poll() is None:  # Timed out
                    print('Timeout')
                    process.terminate()
                    print('Process killed')
                    log.write('!!!!! TIMED OUT !!!!!')
                    all_ok=False
                elif process.returncode != 0:
                    print(f'Error (rc:{process.returncode}), check {log_file}')
                    all_ok=False
                else:
                    print('OK')
                print()
    return all_ok


def parse_logfile(logfile) -> dict:
    """
    Parse a log file and return a dict of stats
    """

    fcontents=[]
    with open(logfile, 'r') as f:
        fcontents = f.readlines()
    stat = {}
    instat = False
    for line in fcontents:
        line = line.strip()
        if line == '':
            continue
        if not instat:
            if '============= MacSim Stats =============' in line:
                instat = True
            continue
        if instat and ':' in line:
            if line[-1] == ':':
                continue
            key, val = line.split(':')
            key, val = key.strip(), val.strip()
            stat[key] = float(val)      
    return stat


def collect_stats(benchmarks:dict, gpu_configs:dict, log_dir:str) -> dict:
    """
    Collect stats from all logs in the log directory and return a dict
    """

    stats = {}
    # Go over all logs
    for gpucfg in gpu_configs.keys(): 
        benchmark_stats = {}
        for benchmark in benchmarks.keys():
            log_file = os.path.join(log_dir, f'{gpucfg}_{benchmark}_run.log')
            # print('Parsing log: {:20s} {:s}'.format('[{:s}]'.format(benchmark), log_file))
            benchmark_stats[benchmark] = parse_logfile(log_file)
        stats[gpucfg] = benchmark_stats
    return stats


def dump_stats(benchmarks:dict, gpu_configs:dict, log_dir:str, stats_to_dump:list, dump_file:str):
    """
    Dump stats to a file as table
    """ 
    
    # Collect all stats
    stats = collect_stats(benchmarks, gpu_configs, log_dir)
    
    txt=''
    for algo in stats.keys():
        # Header
        txt+='\n'+'-'*30+'+'+ ('-'*25+'+')*len(stats_to_dump)+'\n'
        txt+=f'{algo:^30s}|'
        for i, stat in enumerate(stats_to_dump):
            txt+=f'{stat:^25s}|' + ('\n' if i == len(stats_to_dump)-1 else '')
        
        # Bar
        txt+='-'*30+'+'+ ('-'*25+'+')*len(stats_to_dump)+'\n'
            
        # Values
        for bench in benchmarks:
            txt+=f'{bench:30s}|'
            for i, stat in enumerate(stats_to_dump):
                kvpairs = stats[algo][bench]
                txt+=f'{kvpairs[stat]:25}|' + ('\n' if i == len(stats_to_dump)-1 else '')
        # Bar
        txt+='-'*30+'+'+ ('-'*25+'+')*len(stats_to_dump)+'\n'

    with open(dump_file, 'w') as f:
        f.write(txt)
    print(f"Dumped stats to {dump_file}")


def export_json(benchmarks:dict, gpu_configs:dict, log_dir:str, json_file:str):
    """
    Export stat dict as JSON
    """

    # Collect all stats
    stats = collect_stats(benchmarks, gpu_configs, log_dir)

    # export JSON
    json_object = json.dumps(stats, indent=4)
    with open(json_file, "w") as outfile:
        outfile.write(json_object)
    print(f"Exported stats to {json_file}")


if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trace-dir', help='Specify trace directory', type=str, default=os.getenv('MACSIM_TRACE_DIR'))
    parser.add_argument('-l', '--log-dir', help='Specify log directory', type=str, default=os.path.join(os.getenv('MACSIM_DIR'), 'log'))
    parser.add_argument('-c', '--configs', help='Specify xml configs (delimit by :)', type=str)
    parser.add_argument('-b', '--benchmarks', help='Specify benchmarks to run (delimit by :)', type=str)
    parser.add_argument('-r', '--run', help='Run simulator', action='store_true')
    parser.add_argument('-d', '--dump-stats', help='Dump stats to a file', type=str, default='')
    parser.add_argument('-j', '--json', help='Export stats in json format', type=str, default='')
    args = parser.parse_args()

    trace_dir = args.trace_dir
    log_dir = args.log_dir

    # Select configs to run
    gpu_configs={}
    if args.configs:
        configs = args.configs.strip().split(':')
        for config in configs:
            if config in common.gpu_configs.keys():
                gpu_configs[config]=common.gpu_configs[config]
            else:
                print(f'Invalid config {config}')
                exit(1)

    # Select benchmarks to run
    benchmarks = common.benchmarks
    if args.benchmarks:
        benchmark_list = args.benchmarks.strip().split(':')
        benchmarks = {}
        for bench in benchmark_list:
            if bench in common.benchmarks.keys():
                benchmarks[bench] = common.benchmarks[bench]
            else:
                print(f'Invalid benchmark {bench}')
                exit(1)

    try:
        # Run macsim
        if args.run:
            all_ok = run_macsim(benchmarks, gpu_configs, log_dir)
            if not all_ok:
                exit(-1)
        
        # Dump stats in a table format
        if args.dump_stats != '':
            dump_stats(benchmarks, gpu_configs, log_dir, common.stats, args.dump_stats)

        # Dump stats as JSON
        if args.json != '':
            export_json(benchmarks, gpu_configs, log_dir, args.json)

    except Exception as e:
        print(e)