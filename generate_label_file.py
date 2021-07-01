import os
from tqdm import tqdm
data_dir = 'E:/datasets/arithmetic_data/test_labels'
files = os.listdir(data_dir)
label_file = os.path.join('E:/datasets/arithmetic_data', 'gt_test.txt')
with open(label_file, 'w', encoding='utf-8') as f:
    for file in tqdm(files):
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
        image_name = file.replace('txt', 'jpg')
        f.writelines('arithmetic_data/tst_aug_images/' + image_name  + '\t' + lines[0] + '\n')
    
    
    
    # lines = f.read() #readlines()
    # with open('E:/datasets/synthesis_data/a.txt','a', encoding='utf-8') as f1:
    #     f1.write(lines) #writelines()