import os
import subprocess
import shutil
import argparse
from contextlib import contextmanager

def run_command(command):
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        exit(result.returncode)

@contextmanager
def change_directory(target_dir):
    origin_dir = os.getcwd()
    os.chdir(target_dir)
    yield
    os.chdir(origin_dir)

def main():
    #parse arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, help="gemm version")
    parser.add_argument('--ref', type=str, help="ref type, cublas/cutlass")
    args = parser.parse_args()
    print(f"Gemm version:{args.version}")
    print(f"Reference:{args.ref}")

    # s0:remove build 
    build_dir = "build"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        print(f"Removed exsiting '{build_dir}' directory")
    
    # s1:create build
    os.makedirs(build_dir)
    print(f"Created new '{build_dir}' directory")

    # s2:run cmake to gen Makefile
    run_command(f"cmake -DK_VERSION='{args.version}' -DREF_TYPE='{args.ref}' -DSOURCE_FILE=gemm_v'{args.version}'.cu -B '{build_dir}' .")
    print(f"cmake -DK_VERSION='{args.version}' -DREF_TYPE='{args.ref}' -DSOURCE_FILE=gemm_v'{args.version}'.cu -B '{build_dir}' .")
    with change_directory(build_dir):
        run_command("make")
        run_command("./main")
    # s3:compile .cu using nvcc
    '''old_obj_files = [f for f in os.listdir("src") if f.endswith(".o")]
    if old_obj_files:
        run_command("rm -r src/*.o")'''
    #cu_files = [f for f in os.listdir("src") if f.endswith(".cu")]
    '''for cu_file in cu_files:
        run_command(f"nvcc -Iinclude -c src/{cu_file} -o src/{cu_file[:-3]}.o")'''

    # s4:link object files
    #object_files = " ".join([f"src/{cu_file[:-3].o}" for cu_file in cu_files])
    #run_command(f"g++ -std=c++11 -o {build_dir}/main build/main.o {object_files}")


if __name__ == "__main__":
    main()
