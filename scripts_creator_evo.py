detectors = ['yolov3', 'yolov4']
multi_score = [True]
pop_sizes = [50, 70, 90, 110]
weightCLS = [0.1, 0.2]

counter = 0
for detector in detectors:
    for weight in weightCLS:
        for pop in pop_sizes:
            for multi in multi_score:
                f = open(f"./logs/exec_file_{detector}_{counter}", "w")
                a = ""
                a += "#!/bin/bash\n"
                a += "#SBATCH --partition main\n"
                a += "#SBATCH --time 6-00:00:00\n"
                a += "#SBATCH --job-name EvoAttack\n"
                a += f"#SBATCH --output ./logs/{detector}-Adam-FP-CLS-{str(weight)}_pop_{str(pop)}_multi_{multi}-%J.out\n"
                a += "#SBATCH --gpus=rtx_3090:1\n"
                a += "#SBATCH --mail-user=razla@post.bgu.ac.il\n"
                a += "#SBATCH --mail-type=ALL\n"
                a += "#SBATCH --mem=32G\n"
                a += "module load anaconda\n"
                a += "source activate new_xai\n"
                if detector == 'ssd3':
                    if multi:
                        a += f"python ensemble.py --model={detector} --multi  --method=nes --scale=0.25 --opt=adam --attack=fn --epochs=1000 --weightCLS={str(weight)} --pop={str(pop)}"
                    else:
                        a += f"python ensemble.py --model={detector} --method=nes --scale=0.25 --opt=adam --attack=fn --epochs=1000 --weightCLS={str(weight)} --pop={str(pop)}"
                else:
                    if multi:
                        a += f"python ensemble.py --model={detector} --multi --method=nes --tiny --scale=0.25 --opt=adam --attack=fn --epochs=1000 --weightCLS={str(weight)} --pop={str(pop)}"
                    else:
                        a += f"python ensemble.py --model={detector} --tiny --method=nes --scale=0.25 --opt=adam --attack=fn --epochs=1000 --weightCLS={str(weight)} --pop={str(pop)}"

                f.write(a)
                f.close()
                counter += 1
print(counter)