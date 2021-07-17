### test pipe file to run all DNA simulations from one source

import subprocess
import sys
import os
import numpy as np
import shutil, glob

## write the following as = True or False
## if traj already computed fill False

temp_list = [326]
step_num = 100000
reps = 1
compute_traj = True
fix_random = False
rand_seed = 12345
export_pngs = True
box = '36'
prefix = 'vary_box_' + box + '-'
main_file = "main_no-v_200-traj-unwrap.in"
conf_lammps_file = 'conf_lammps-' + box + '.in'
run_reps = 1
#conf_lammps_file = 'conf_lammps.in'

png_dir_name = "png_export" + '-' + prefix

## locations by line number in main run script:

temp_line = 7
rand_var_line = 11
conf_file_line = 27
run_line = 78

## should not need to edit below this line

## will substitute following variables if exactly 8 args are given

print(len(sys.argv))
print(sys.argv)

if len(sys.argv) == 8:
    temp_list = [int([sys.argv[1]][0])]
    step_num = int(sys.argv[2])
    reps = int(sys.argv[3])
    conf_lammps_file = sys.argv[4]
    prefix = sys.argv[5]
    main_file = sys.argv[6]
    run_reps = sys.argv[7]
    
print (sys.argv)
print (len(sys.argv))
print (temp_list, step_num, reps)

dir_list = []

for temp in temp_list:

    with open(main_file, 'r') as file:

        data = file.readlines()
        data[temp_line] = "variable T equal " + str(temp) + "\n"
        data[conf_file_line] = "read_data " + conf_lammps_file + "\n"
        
        # data[run_line] = "run " + str(step_num) + "\n"
        
        ## writes new run lines for a given number of run_reps
        for line_num in range(int(run_reps)):
            
            if (run_line + line_num) < len(data):
                data[run_line + line_num] = "run " + str(step_num) + "\n"
                
            else:
                data.append("run " + str(step_num) + "\n")
        
        
        if fix_random:
            data[rand_var_line] = "variable random equal " + str(rand_seed) + "\n"
            
        # for formatting output filename:
        temp = "T-" + str(temp)
        times = data[52].replace("timestep ", "").replace(".0", "").replace("\n", "")
        times = "t-" + str(times)
        steps = np.format_float_scientific(float(data[78].replace("run ", ""))).replace(".","")
        steps = "s-" + str(steps)

    # saves main files into directory to prevent interference on main.in
    main_save = "main_dir/" + main_file.replace(".in", "-" + prefix + str(temp) + ".in")
    with open(main_save, 'w') as file: 
        file.writelines(data)

    save_name_outer = prefix + '_' + temp + '_' + times + '_' + steps
    
    subdirs = [x[0].replace(".", "") for x in os.walk('.')]
    
    for rep in range(reps):
        
        if not fix_random:
            with open(main_save, 'r') as file:
                data[11] = "variable random equal " + str(np.random.randint(1, 100000)) + "\n"
            with open(main_save, 'w') as file: 
                file.writelines(data)
                
        if reps > 1:
            sub_dir = "/rep_" + str(rep)
            subprocess.call(['mkdir', save_name_outer])  ## new line added to fix reps bug
            save_name = save_name_outer + sub_dir
        else:
            save_name = save_name_outer
 
        print (save_name)
        dir_list.append(save_name)

        if compute_traj:
            subprocess.call(['mkdir', save_name])
            subprocess.call(['cp', main_save, save_name + '/' + main_file])
            subprocess.call(['cp', conf_lammps_file, save_name + '/' + conf_lammps_file])
            #subprocess.call(['cp', 'in00_cvmd.psf', save_name + '/in00_cvmd.psf'])
        subprocess.call(['cp', 'plot_EBP.py', save_name + '/plot_EBP.py'])
        
def comp_traj(reps):
    if reps > 1:
        subprocess.call(['../../../../../lmp_serial', '-in', main_file])
    else:
        subprocess.call(['../../../../lmp_serial', '-in', main_file])
        
def plot_basepairs(filename='plot_EBP.py'):
    subprocess.call(['python', filename])
    
def export_pngs(reps, dir_name=png_dir_name):
    cwd = os.getcwd()
    subprocess.call(['ls'])
    #subprocess.call(['cp', '*.png', '../../' + png_dir_name])
    for png in glob.iglob(os.path.join(cwd, "*.png")):
        if os.path.isfile(png):
            if reps > 1:
                shutil.copy(png, '../../' + png_dir_name + png.replace(cwd, ""))
            else:
                shutil.copy(png, '../' + png_dir_name + png.replace(cwd, ""))
                
cwd = os.getcwd() 
if export_pngs:
    subprocess.call(['mkdir', png_dir_name])
    
## iterates through each directory to compute, plot and export (if each is desired)
for dir_name in dir_list:
   
    os.chdir(cwd + "/" + dir_name + "/")
    
    if compute_traj:
        comp_traj(reps)
    plot_basepairs()
    if export_pngs:
        export_pngs(reps)
        
# copy and remove batch files from main runfolder directory


