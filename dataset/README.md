# Dataset record

The training notebooks build a synthetic YOLO object-detection dataset from
cropped traffic-sign images pasted onto background photographs. The generated
dataset used in the recorded training run contains **1,500 images**.

## Source and access

The notebooks use the following sources:

1. **Traffic-sign crops**
   - Tuan AI, [`tuanai/traffic-signs-dataset`](https://www.kaggle.com/datasets/tuanai/traffic-signs-dataset)
   - The first training notebook uses these classification-only cropped signs.
   - The later `test_2_robot_again.ipynb` notebook also uses
     [`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
     and adds Tuan AI crops for four classes.
2. **Background photographs**
   - MIT Indoor Scenes, Quattoni and Torralba, obtained through the Kaggle
     copy [`itsahmad/indoor-scenes-cvpr-2019`](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019).
   - The recorded run saved 199 usable MIT background images. The notebook
     reports 200 before one unusable image was removed.

The raw datasets and generated images are not stored in this repository.
The notebooks download the source datasets with `kagglehub`; Kaggle access
credentials must be supplied locally and must never be committed.

## Dataset summary

| Item | Value |
|---|---|
| Target classes | `no_entry`, `go_left`, `go_right`, `go_straight`, `zebra_crossing` |
| Training samples | 1,050 images |
| Validation samples | 300 images |
| Testing samples | 150 images |
| Total samples | 1,500 images |
| Split ratio | 70% train / 20% validation / 10% test |
| Annotation format | YOLO object-detection format: one `.txt` label per image with `class_id`, normalized center `x/y`, normalized width, and normalized height |
| Image size used for training | 640 pixels |

The source crop counts recorded by the first notebook are:

| Class | Crop images |
|---|---:|
| `no_entry` | 210 |
| `go_left` | 31 |
| `go_right` | 108 |
| `go_straight` | 28 |
| `zebra_crossing` | 189 |

The later GTSRB-based notebook records the corresponding source counts as
2,220, 840, 1,378, 2,400, and 480 respectively before compositing.

## Preparation and reproducibility

The pipeline:

1. Selects the five target sign classes from the source crop dataset.
2. Collects background images, primarily from MIT Indoor Scenes.
3. Splits crops and backgrounds into train, validation, and test groups before
   compositing to prevent split leakage.
4. Pastes signs onto backgrounds and writes bounding boxes in YOLO format.
5. Trains an Ultralytics YOLOv8n detector using `dataset/data.yaml`.

The generated split counts and source details above are recorded in the
outputs of [`test_2_robot.ipynb`](../src/training/test_2_robot.ipynb) and
[`test_2_robot_again.ipynb`](../src/training/test_2_robot_again.ipynb).
