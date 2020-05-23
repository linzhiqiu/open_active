if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--script_file') # Where the log files will be saved
    parser.add_argument('--parallel_max', default=6, type=int, help='The max number of job to run in parallel.')
    args = parser.parse_args()

    if not os.path.exists(args.script_file):
        print(args.script_file+" do not exist.")
    
    
    script_dir = args.script_file[:-3]
    if not os.path.exists(script_dir) : os.makedirs(script_dir)
    exps = list(map(lambda x : x.strip(), open(args.script_file, "r").readlines()))
    each_file_lines = int(len(exps) / args.parallel_max)
    sub_script_files = [exps[i:i + each_file_lines] for i in range(0, len(exps), each_file_lines)]
    script_parallel = os.path.join(script_dir, "parallel.sh")
    with open(script_parallel, "w+") as parallel_file:
        parallel_file.write("#!/bin/sh \n")
        for i, sub_script_file in enumerate(sub_script_files):
            sub_script_file_path = os.path.join(script_dir,str(i)+".sh")
            with open(sub_script_file_path, "w+") as file:
                file.write("#!/bin/sh \n")
                for s in sub_script_file:
                    file.write("bash -c \"" + s +"\" ; \n")
            parallel_file.write("sh "+sub_script_file_path+ " & ;\n")
    print(f"Parallel script saved at {script_parallel}")  
            
