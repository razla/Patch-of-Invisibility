## ensemble.py records:
1. Best **YOLOv3-tiny** ensemble.py params:

    ```Namespace(seed=15089, model='yolov3', dataset='one_person', tiny=True, method='bbgan', attack='fn', epochs=1000, batch=8, lr=0.02, opt='adam', scale=0.25, multi=True, weightTV=0.1, weightCLS=0.1, classGAN=259, classDET=0, classTGT=1, pop=110, sigma=0.1)```

2. Best **YOLOv4-tiny** ensemble.py params:

    ```Namespace(seed=15089, model='yolov4', dataset='one_person', tiny=True, method='bbgan', attack='fn', epochs=1000, batch=8, lr=0.02, opt='adam', scale=0.25, multi=True, weightTV=0.1, weightCLS=0.1, classGAN=259, classDET=0, classTGT=1, pop=70, sigma=0.1)```

3. Best **YOLOv5-tiny** ensemble.py params:

    ***Note** that the best YOLOv5 seed from the paper is unknown, but we found a new good enough seed=9*

    ```Namespace(seed=9, model='yolov4', dataset='one_person', tiny=True, method='bbgan', attack='fn', epochs=1000, batch=8, lr=0.02, opt='adam', scale=0.25, multi=True, weightTV=0.1, weightCLS=0.1, classGAN=259, classDET=0, classTGT=1, pop=90, sigma=0.1)```