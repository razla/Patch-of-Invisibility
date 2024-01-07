import matplotlib.pyplot as plt
import numpy as np
import os

LOGS_FULL_PATH = '/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo4/multi-false'

def total_loss_plot(total_losses, exp_names):
    for i, total_loss in enumerate(total_losses):
        plt.plot(np.arange(len(total_loss)), total_loss, label=exp_names[i])

    print(np.array([total_losses[i][-1] for i in range(len(total_losses))]).mean())

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.ylim(0.35, 1)

    if 'yolo3' in LOGS_FULL_PATH:
        if 'multi-false' in LOGS_FULL_PATH:
            plt.title(r'Tiny-YOLOv3, Obj only')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo3-multi-false-total.png')
        else:
            plt.title(r'Tiny-YOLOv3, $Obj \times det$')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo3-multi-true-total.png')
    else:
        if 'multi-false' in LOGS_FULL_PATH:
            plt.title(r'Tiny-YOLOv4, Obj only')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo4-multi-false-total.png')
        else:
            plt.title(r'Tiny-YOLOv4, $Obj \times det$')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo4-multi-true-total.png')

def det_loss_plot(det_losses, exp_names):
    for i, det_loss in enumerate(det_losses):
        plt.plot(np.arange(len(det_loss)), det_loss, label=exp_names[i])

    plt.ylabel('Loss')
    plt.xlabel('Epochs')

    if 'yolo3' in LOGS_FULL_PATH:
        if 'multi-false' in LOGS_FULL_PATH:
            plt.title(r'Tiny-YOLOv3, Obj only')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo3-multi-false-det.png')
        else:
            plt.title(r'Tiny-YOLOv3, $Obj \times det$')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo3-multi-true-det.png')
    else:
        if 'multi-false' in LOGS_FULL_PATH:
            plt.title(r'Tiny-YOLOv4, Obj only')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo4-multi-false-det.png')
        else:
            plt.title(r'Tiny-YOLOv3, $Obj \times det$')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo4-multi-true-det.png')

def cls_loss_plot(cls_losses, exp_names):
    for i, cls_loss in enumerate(cls_losses):
        plt.plot(np.arange(len(cls_loss)), cls_loss, label=exp_names[i])

    plt.ylabel('Loss')
    plt.xlabel('Epochs')

    if 'yolo3' in LOGS_FULL_PATH:
        if 'multi-false' in LOGS_FULL_PATH:
            plt.title(r'Tiny-YOLOv3, Obj only')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo3-multi-false-cls.png')
        else:
            plt.title(r'Tiny-YOLOv3, $Obj \times det$')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo3-multi-true-cls.png')
    else:
        if 'multi-false' in LOGS_FULL_PATH:
            plt.title(r'Tiny-YOLOv4, Obj only')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo4-multi-false-cls.png')
        else:
            plt.title(r'Tiny-YOLOv3, $Obj \times det$')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo4-multi-true-cls.png')

def tv_loss_plot(tv_losses, exp_names):
    for i, tv_loss in enumerate(tv_losses):
        plt.plot(np.arange(len(tv_loss)), tv_loss, label=exp_names[i])

    plt.ylabel('Loss')
    plt.xlabel('Epochs')

    if 'yolo3' in LOGS_FULL_PATH:
        if 'multi-false' in LOGS_FULL_PATH:
            plt.title(r'Tiny-YOLOv3, Obj only')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo3-multi-false-tv.png')
        else:
            plt.title(r'Tiny-YOLOv3, $Obj \times det$')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo3-multi-true-tv.png')
    else:
        if 'multi-false' in LOGS_FULL_PATH:
            plt.title(r'Tiny-YOLOv4, Obj only')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo4-multi-false-tv.png')
        else:
            plt.title(r'Tiny-YOLOv3, $Obj \times det$')
            plt.legend()
            plt.savefig('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/analysis_logs/yolo4-multi-true-tv.png')


def translate_exp_name(name):
    exp_name = name.split('-')[1].split('_')
    cls = exp_name[0]
    if exp_name[-1] == '':
        pop_size = exp_name[-2]
    else:
        pop_size = exp_name[-1]
    if pop_size == '99':
        pop_size = '110'
    new_exp_name = f'N: {pop_size}, '
    new_exp_name += r'$\lambda_{cls}$: ' + f'{cls}'
    return new_exp_name

# Function to read
# last N lines of the file
def LastNlines(fname):
    # open the sample file used
    file = open(fname, 'r')

    # read the content of the file opened
    content = reversed(file.readlines())
    total_loss = None
    cls_loss = None
    tv_loss = None
    det_loss = None
    for i, word in enumerate(content):
        if 'Total loss:' in word:
            total_loss = i
            break

    for i, word in enumerate(content):
        if 'Total loss cls:' in word:
            cls_loss = i
            break

    for i, word in enumerate(content):
        if 'Total loss tv:' in word:
            tv_loss = i
            break

    for i, word in enumerate(content):
        if 'Total loss det:' in word:
            det_loss = i
            break

    file.close()
    file = open(fname, 'r')
    new_content = file.readlines()
    det_loss_text = new_content[-det_loss:]
    new_det_loss_text = ''
    for content in det_loss_text:
        new_det_loss_text += content.replace('\n', '').replace('[', '').replace(']','')

    file.close()
    file = open(fname, 'r')
    new_content = file.readlines()
    tv_loss_text = new_content[-tv_loss:]
    new_tv_loss_text = ''
    for content in tv_loss_text:
        new_tv_loss_text += content.replace('\n', '').replace('[', '').replace(']','')

    file.close()
    file = open(fname, 'r')
    new_content = file.readlines()
    cls_loss_text = new_content[-cls_loss:]
    new_cls_loss_text = ''
    for content in cls_loss_text:
        new_cls_loss_text += content.replace('\n', '').replace('[', '').replace(']','')

    file.close()
    file = open(fname, 'r')
    new_content = file.readlines()
    total_loss_text = new_content[-cls_loss:]
    new_total_loss_text = ''
    for content in total_loss_text:
        new_total_loss_text += content.replace('\n', '').replace('[', '').replace(']','')

    new_total_loss_text_list = new_total_loss_text.split(' ')
    total_loss_list = []
    for number in new_total_loss_text_list:
        if number != '':
            total_loss_list.append(float(number))

    new_cls_loss_text_list = new_cls_loss_text.split(' ')
    cls_loss_list = []
    for number in new_cls_loss_text_list:
        if number != '':
            cls_loss_list.append(float(number))

    new_tv_loss_text_list = new_tv_loss_text.split(' ')
    tv_loss_list = []
    for number in new_tv_loss_text_list:
        if number != '':
            tv_loss_list.append(float(number))

    new_det_loss_text_list = new_det_loss_text.split(' ')
    det_loss_list = []
    for number in new_det_loss_text_list:
        if number != '':
            det_loss_list.append(float(number))

    total_loss = np.array(total_loss_list)
    cls_loss = np.array(cls_loss_list)
    tv_loss = np.array(tv_loss_list)
    det_loss = np.array(det_loss_list)

    return total_loss, cls_loss, tv_loss, det_loss

# Driver Code:
if __name__ == '__main__':
    exp_names = []
    total_losses = []
    cls_losses = []
    tv_losses = []
    det_losses = []
    files = sorted(os.listdir(LOGS_FULL_PATH))
    for file in files:
        full_path = os.path.join(LOGS_FULL_PATH, file)
        exp_name = file.split('.out')[0][15:-18]
        if '.DS_Store' in full_path:
            continue
        exp_name = translate_exp_name(exp_name)
        exp_names.append(exp_name)
        try:
            total_loss, cls_loss, tv_loss, det_loss = LastNlines(full_path)
            total_losses.append(total_loss)
            cls_losses.append(cls_loss)
            tv_losses.append(tv_loss)
            det_losses.append(det_loss)
        except:
            print(f'{file} not found')

    total_loss_plot(total_losses, exp_names)
    # cls_loss_plot(cls_losses, exp_names)
    # tv_loss_plot(tv_losses, exp_names)
    # det_loss_plot(det_losses, exp_names)