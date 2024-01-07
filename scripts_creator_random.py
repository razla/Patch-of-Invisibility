detectors = ['yolov3', 'yolov4']
multi_score = [True]
epochs = [50000, 70000, 90000, 110000]

counter = 0
for detector in detectors:
    for epoch in epochs:
        for multi in multi_score:
            f = open(f"./exec_file_{detector}_{counter}_random_gan", "w")
            a = ""
            a += "#!/bin/bash\n"
            a += "#SBATCH --partition main\n"
            a += "#SBATCH --time 6-00:00:00\n"
            a += "#SBATCH --job-name EvoAttack\n"
            a += f"#SBATCH --output ./logs/{detector}-Adam-FP-RandomGAN_multi_{multi}-%J.out\n"
            # a += "#SBATCH --gpus=rtx_3090:1\n"
            a += "#SBATCH --gpus=1\n"
            a += "#SBATCH --mail-user=razla@post.bgu.ac.il\n"
            a += "#SBATCH --mail-type=ALL\n"
            a += "#SBATCH --mem=32G\n"
            a += "module load anaconda\n"
            a += "source activate new_xai\n"
            if multi:
                a += f"python ensemble.py --model={detector} --multi --method=random_gan --tiny --scale=0.25 --opt=adam --attack=fn --epochs={epoch}"
            else:
                a += f"python ensemble.py --model={detector} --tiny --method=random_gan --scale=0.25 --opt=adam --attack=fn --epochs={epoch}"

            f.write(a)
            f.close()
            counter += 1
print(counter)