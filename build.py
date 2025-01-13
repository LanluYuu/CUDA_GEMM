import os
import subprocess
import shutil

def run_command(command):
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        exit(result.returncode)

def main():
    # s0:remove build 
    build_dir = "build"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        print(f"Removed exsiting '{build_dir}' directory")
    
    # s1:create build
    os.makedirs(build_dir)
    print(f"Created new '{build_dir}' directory")

    # s2:run cmake to gen Makefile
    run_command(f"cmake -B '{build_dir}' .")

    run_command(f"cd {build_dir} | make")
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

    # s5 execute 
    run_command(f"./main")
    run_command("cd ../")

if __name__ == "__main__":
    main()
