import numpy as np
import shutil
import os

path = "/home/razla/AttaXAI/src/batches"

def filter_output_func(dir):
    if "output" in dir:
        return True
    return False

def filter_adv_func(dir):
    if "best_adv" in dir:
        return True
    return False


output_dirs = filter(filter_output_func, os.listdir(path))

for output_dir in output_dirs:
    print(output_dir)
    full_path = os.path.join(path, output_dir)
    exps_paths = os.listdir(full_path)
    for exp_path in exps_paths:
        full_exp_path = os.path.join(full_path, exp_path)
        images_full_exp_path = os.listdir(full_exp_path)
        rand_imgs = np.random.choice(images_full_exp_path, 10)
        print(rand_imgs)
        for rand_img in rand_imgs:
            img_path = os.path.join(full_exp_path, str(rand_img))
            src_img_path = os.path.join(img_path, os.listdir(img_path)[0])
            src_img_name = os.listdir(img_path)[0]
            target_img_path = os.listdir(src_img_path)[0]
            target_img_name = os.listdir(src_img_path)[0]

            inner_files = os.listdir(os.path.join(src_img_path, target_img_path))

            best_adv_img = next(iter(filter(filter_adv_func, inner_files)))

            src_file = os.path.join(src_img_path, target_img_path, best_adv_img)
            dest_file = os.path.join("/home/razla/attaxai_examples", exp_path, f'{src_img_name}-{target_img_name}.png')
            if not os.path.exists(os.path.join("/home/razla/attaxai_examples", exp_path)):
                os.makedirs(os.path.join("/home/razla/attaxai_examples", exp_path))
            shutil.copy(src_file, dest_file)
