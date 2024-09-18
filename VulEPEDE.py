import argparse
import pandas as pd
import pickle
import os
import glob
import torch
from tqdm import tqdm

from epede_model import Classifier
from data_loader import CustomDataset
from torch.utils.data import random_split


def parse_options():
    parser = argparse.ArgumentParser(description='model training.')
    parser.add_argument('-i', '--input', help='The dir path', type=str, required=True)
    parser.add_argument('-o', '--output', help='The result path', type=str, required=True)
    parser.add_argument('-d', '--device', help='gpu number', default='cuda:0')
    parser.add_argument('-m', '--model_name', help='The model name', default='ffmpegqemu')
    args = parser.parse_args()
    return args


def load_data(filename):
    # print("Begin to load data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def generate_dataframe(input_path):
    input_path = input_path + "/" if input_path[-1] != "/" else input_path
    data_list = []
    for type_name in tqdm(os.listdir(input_path)):
        dic_name = input_path + type_name
        filename = glob.glob(dic_name + "/*.pkl")
        for file in filename:
            
            data = load_data(file)
            data_list.append({
                "filename": file.split("/")[-1].rstrip(".pkl"),  
                "pdg": data[0],
                "cddf": data[1],
                "label": 0 if type_name == "No-Vul" else 1  
            })
    
    final_dic = pd.DataFrame(data_list)
    return final_dic


def main():
    args = parse_options()
    data_path = args.input
    device = args.device
    out_path = args.output
    model_name = args.model_name
    print('Begin to load data!')
    data_df = generate_dataframe(data_path)
    print('Finish load data!')
    filename = data_df['filename']
    pdg = data_df['pdg']
    cddf = data_df['cddf']
    label = data_df['label']
    dataset = CustomDataset(filename=filename, pdg=pdg, cddf=cddf, labels=label, pdg_max_len=64, cddf_max_len=128, cddf_dim=64)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print("train_size: ", train_size)
    print("val_size: ", val_size)
    print("test_size: ", test_size)
    
    max_len = 64
    batch_size = 64
    learning_rate = 0.001
    result_save_path = out_path
    epochs = 100
    torch.manual_seed(4)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])   
    classifier = Classifier(device=device, model_name=model_name,
                                max_len=max_len,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                result_save_path=result_save_path,
                                epochs=epochs)
    
    classifier.preparation(train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset)
    
    classifier.train()
    

if __name__ == "__main__":
    main()

