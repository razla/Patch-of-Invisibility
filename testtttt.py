import count_map.main as eval_map

def check():
    output_map = eval_map.count(path_ground_truth="/cs_storage/razla/NatAdvPatcj/eval_output/yolov3tiny_temp_f/output_imgs/yolo-labels-rescale_yolov3Truetiny",
                                path_detection_results="/cs_storage/razla/NatAdvPatcj/eval_output/yolov3tiny_temp_f/output_imgs/yolo-labels-rescale_yolov3Falsetiny",
                                path_images_optional=None)

    with open("/cs_storage/razla/NatAdvPatcj/wtf/map.txt", "w") as text_file:
        text_file.write(str(output_map))

if __name__ == '__main__':
    check()