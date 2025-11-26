import os

pred_dir = "/home/omar-mohamed/runs/final prerdict/kitti_yolov8m_only_test_predict/labels"
num_images = 7517

for i in range(num_images):
    filename = f"{i:06d}.txt"
    path = os.path.join(pred_dir, filename)
    if not os.path.exists(path):
        open(path, 'w').close()  # create empty file
