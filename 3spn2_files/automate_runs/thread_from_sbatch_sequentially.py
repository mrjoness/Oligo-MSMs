## take an sbatch command and pass it in as multiple processes/threads to pipe_vary_temp

import subprocess
import sys
import os
from multiprocessing import Pool
    
process_num = int(sys.argv[1])
steps = sys.argv[2]
reps = sys.argv[3]
conf_lammps_file = sys.argv[4]
prefix = sys.argv[5]
main_file = sys.argv[6]
start_temp = sys.argv[7]
temp_reps = sys.argv[8]
temp_index = sys.argv[9]
run_reps = sys.argv[10]

temp = start_temp
counter = 0

def get_temp():
    
    global counter
    
    #temp = int(start_temp) + int((counter + int(temp_index)) / int(temp_reps))
    counter += 1
    
    print("counter = ", counter)
    return str(temp)

def new_run(run_num, temp=temp, steps=steps, reps=reps, conf_lammps_file=conf_lammps_file, prefix=prefix, main_file=main_file, run_reps=run_reps):
    
    prefix += ("-" + str(run_num))
    subprocess.call(["python", "pipe_vary_temp.py", temp, steps, reps, conf_lammps_file, prefix, main_file, run_reps])

process_num_list = [i for i in range(process_num)]

if __name__ == '__main__':
    p = Pool(process_num)
    p.map(new_run, process_num_list)

else:
    print ("not main")

# sample output: python pipe_vary_temp.py 300 20000 1 conf_AT-all.in repeat-test-login main_no-v_200-traj-unwrap.in 3