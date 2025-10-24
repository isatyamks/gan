import torch.nn as nn
import csv
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def init_csv(out_dir):
    csv_path = os.path.join(out_dir, "training_log.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Discriminator_Loss', 'Generator_Loss', 'Batch_Size', 'LR', 'Beta1', 'ngf', 'ndf', 'nz'])
    csv_file.flush()
    return csv_file, csv_writer
