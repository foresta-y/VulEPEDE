import os
import lap
import torch
import numpy
import pickle
import numpy as np
from sklearn.metrics._classification import _check_set_wise_labels
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
import time

from cnn_module import CBAM


def sava_data(filename, data):
    print("Begin to save data：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()


def get_accuracy(labels, prediction):
    cm = confusion_matrix(labels, prediction)

    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    def linear_assignment(cost_matrix):
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy


def calculate_binary_classification_metrics(labels, predictions):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 计算精确率、召回率、F1分数
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    # 计算假阴性率和假阳性率
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    
    # 返回结果
    return {
        'accuracy': format(accuracy * 100, '.3f'),
        'precision': format(precision * 100, '.3f'),
        'recall': format(recall * 100, '.3f'),
        'f1_score': format(f1_score * 100, '.3f'),
        'false_negative_rate': format(fnr * 100, '.3f'),
        'false_positive_rate': format(fpr * 100, '.3f')
    }


# 位置编码
class PositionalEncoding(nn.Module):
    # "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.size(1), :].to(x.device) 
        return self.dropout(pe)
    

class CrossAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.query_linear = torch.nn.Linear(dim, dim)
        self.key_linear = torch.nn.Linear(dim, dim)
        self.value_linear = torch.nn.Linear(dim, dim)

        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, feature_a, feature_b):
        # Linear transformations
        Q = self.query_linear(feature_a)  # (batch_size, seq_len_a, dim)
        K = self.key_linear(feature_b)    # (batch_size, seq_len_b, dim)
        V = self.value_linear(feature_b)  # (batch_size, seq_len_b, dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V)

        return output


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, num_values, num_heads, ff_dim, num_layers, dropout_rate, max_len):
        super(TransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_values = num_values
        self.positional_encoding = PositionalEncoding(self.embedding_dim - self.num_values, 0.1, max_len)
        # self.positional_encoding = PositionalEncoding(self.embedding_dim, 0.1, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.embedding_dim * max_len, self.embedding_dim)

    def forward(self, x):
        # 获取位置编码
        pe = self.positional_encoding(x)
        # 拼接特征的后8个维度与位置编码
        x_last_8 = x[:, :, -self.num_values:]  # 提取特征的后8个维度
        x_rest = x[:, :, :-self.num_values]    # 提取除去后8个维度的sent2vec嵌入
        pe_expanded = pe.expand(x.size(0), -1, -1)  # 扩展位置编码的批次维度
        pe_combined = torch.cat((x_last_8, pe_expanded), dim=-1)  # EPE
        # 添加拼接后的结果到原嵌入上
        x = x_rest + pe_combined
        # x = x_rest + pe
        # Transformer 编码器
        x = x.transpose(0, 1)  # (num_sentences, batch_size, embedding_dim + num_values)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, num_sentences, embedding_dim + num_values)
        # 展平并通过全连接层
        x = x.reshape(x.size(0), -1)  # 展平为 (batch_size, num_sentences * embedding_dim)
        x = self.fc(x)  # 全连接层
        return x


class ConvNet(nn.Module):
    def __init__(self, in_channel, out_dim):
        super(ConvNet, self).__init__()
        self.in_channel = in_channel
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1)
        self.cbam1 = CBAM(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.cbam2 = CBAM(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 初始化全连接层，但不指定输入特征的数量
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(128 * 14 * 14, self.out_dim) 

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x = F.relu(self.cbam1(self.conv1(x)))
        x = self.pool(x)
        # x = F.relu(self.conv2(x))
        x = F.relu(self.cbam2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        # # 在这里动态计算全连接层的输入特征数量
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x
    
    
class VulEPEDENN(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super(VulEPEDENN, self).__init__()
        self.hidden_size = hidden_size
        self.cddf_nn = ConvNet(2, hidden_size)
        self.vsdg_nn = TransformerModel(embedding_dim=hidden_size, 
                                        num_values=8, num_heads=4, 
                                        ff_dim=hidden_size, num_layers=4,  
                                        dropout_rate=0.1, max_len=64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        
    def forward(self, x):
        pdg, cddf = x
        pdg_out = self.vsdg_nn(pdg)
        cddf_out = self.cddf_nn(cddf)
        out_ = torch.cat((pdg_out, cddf_out), dim=1)
        out_ = self.relu(out_)
        out = self.fc(out_)
        return out_, out


class Classifier():
    def __init__(self, device='cuda:0', model_name='VulEPEDE', max_len=64, epochs=100, batch_size=32, learning_rate=0.01, result_save_path="/root/data/qm_data/vulcnn/data/results"):
        self.model_name = model_name
        self.model_saved_path = './model/' + model_name + '_trained_best_f1_model.pt'
        self.model = VulEPEDENN(hidden_size=max_len, output_dim=2)
        self.device = device
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(torch.device(self.device))
        result_save_path = result_save_path + "/" if result_save_path[-1] != "/" else result_save_path
        if not os.path.exists(result_save_path): os.makedirs(result_save_path)
        self.result_save_path = result_save_path + "_epo" + str(epochs) + "_bat" + str(batch_size) + ".result"

    def preparation(self, train_dataset, valid_dataset, test_dataset):
        # create data loaders
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # helpers initialization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        
        
    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            filename, pdg, cddf, targets = data
            pdg = pdg.to(torch.device(self.device))
            cddf = cddf.to(torch.device(self.device))
            targets = targets.to(torch.device(self.device))
            with autocast():
                _out, outputs = self.model([pdg, cddf])
                loss = self.loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()
            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))
            labels += list(np.array(targets.cpu()))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}')
        train_loss = np.mean(losses)
        score_dict = calculate_binary_classification_metrics(labels, predictions)
        return train_loss, score_dict

    def valid(self):
        print("start validating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        with torch.no_grad():
            for _, data in progress_bar:
                filename, pdg, cddf, targets = data
                pdg = pdg.to(torch.device(self.device))
                cddf = cddf.to(torch.device(self.device))
                targets = targets.to(torch.device(self.device))
                _out, outputs = self.model([pdg, cddf])
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                losses.append(loss.item())
                progress_bar.set_description(
                    f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}')
                
        score_dict = calculate_binary_classification_metrics(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict

    def test(self):
        self.trained_model = VulEPEDENN(hidden_size=self.max_len, output_dim=2)
        self.trained_model.load_state_dict(torch.load(self.model_saved_path))
        self.trained_model.to(torch.device(self.device))
        self.trained_model.eval()
        all_embeddings = []
        all_preds = []
        pre = []
        label = []
        all_y_scores = []
        all_y_true = []
        # progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                filename, pdg, cddf, targets = data
                pdg = pdg.to(torch.device(self.device))
                cddf = cddf.to(torch.device(self.device))
                targets = targets.to(torch.device(self.device))
                _out, outputs = self.trained_model([pdg, cddf])
                preds = torch.argmax(outputs, dim=1).flatten()
                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                
                probabilities = F.softmax(outputs, dim=1)
                y_score = probabilities[:, 1].cpu().numpy()
                y_true = targets.cpu().numpy()
                all_y_scores.extend(y_score)
                all_y_true.extend(y_true)
                
                all_embeddings.append(_out.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                
        all_embeddings = np.vstack(all_embeddings)
        all_preds = np.hstack(all_preds)
        with open('./analyze/tsne/embeddings.pkl', 'wb') as f:
            pickle.dump((all_embeddings, all_preds), f)
        
        with open('./analyze/roc_auc.pkl', 'wb') as f:
            pickle.dump((all_y_scores, all_y_true), f)
        
        score_dict = calculate_binary_classification_metrics(label, pre)
        # val_loss = np.mean(losses)
        return score_dict


    def train(self):
        learning_record_dict = {}
        valid_f1 = 0.0
        train_table = PrettyTable(['type', 'epoch', 'loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'FNR', 'FPR', 'time'])
        valid_table = PrettyTable(['type', 'epoch', 'loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'FNR', 'FPR', 'time'])
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            t = time.perf_counter()
            train_loss, train_score = self.fit()
            train_time = time.perf_counter() - t
            train_table.add_row(["tra", str(epoch + 1), format(train_loss, '.4f')] + [train_score[j] for j in train_score] + [format(train_time, '.4f')])
            print(train_table)
            t2 = time.perf_counter()
            val_loss, val_score = self.valid()
            val_time = time.perf_counter() -t2
            valid_table.add_row(["valid", str(epoch + 1), format(val_loss, '.4f')] + [val_score[j] for j in val_score] + [format(val_time, '.4f')])
            print(valid_table)
            print("\n")
            
            learning_record_dict[epoch] = {'train_loss': train_loss, 'valid_loss': val_loss, "train_score": train_score, "valid_score": val_score}
            sava_data(self.result_save_path, learning_record_dict)
            print("\n")

            if float(val_score['f1_score']) > valid_f1:
                valid_f1 = float(val_score['f1_score'])
                torch.save(self.model.state_dict(), self.model_saved_path)
        
        test_score = self.test()
        print("Test score:")
        for key in test_score:
            print(key, ":", test_score[key])
        
        torch.save(self.model.state_dict(), './model/' + self.model_name + '_trained_model_last.pt')
        
        