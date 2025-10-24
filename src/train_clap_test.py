import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from transformers import ClapModel, ClapProcessor, get_scheduler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# -------------------------- 1. 数据集类定义 --------------------------
class EATDCorpusDataset(Dataset):
    """加载EATD-Corpus数据集，处理音频和标签"""

    def __init__(self, data_root, volunteer_ids, target_sample_rate=48000, depression_threshold=53):
        self.data_root = data_root
        self.volunteer_ids = volunteer_ids
        self.threshold = depression_threshold  # 抑郁判断阈值
        self.target_sample_rate = target_sample_rate  # 目标采样率
        self.resamplers = {}  # 缓存重采样器（避免重复创建）
        self.samples = []  # 存储（音频路径，标签）

        self._load_samples()  # 加载样本
        print(f"成功加载 {len(self.samples)} 个音频样本")

    def _load_samples(self):
        """递归加载志愿者音频和标签"""
        for volunteer_id in self.volunteer_ids:
            volunteer_folder = os.path.join(self.data_root, volunteer_id)
            if not os.path.isdir(volunteer_folder):
                continue  # 跳过非目录

            # 读取标签文件（SDS分数）
            label_file = os.path.join(volunteer_folder, 'new_label.txt')
            if not os.path.exists(label_file):
                continue  # 跳过无标签文件

            with open(label_file, 'r') as f:
                sds_score = float(f.read().strip())
            label = 1 if sds_score >= self.threshold else 0  # 1=抑郁，0=非抑郁

            # 加载positive/negative/neutral三种音频
            for audio_type in ['positive', 'negative', 'neutral']:
                audio_file = f'{audio_type}_out.wav'
                audio_path = os.path.join(volunteer_folder, audio_file)
                if os.path.exists(audio_path):
                    # 过滤空文件或损坏文件（大小>1KB）
                    try:
                        if os.path.getsize(audio_path) > 1024:
                            self.samples.append((audio_path, label))
                        else:
                            print(f"警告：跳过空文件 {audio_path}")
                    except OSError as e:
                        print(f"警告：无法访问文件 {audio_path}，错误：{e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回处理后的音频波形、采样率和标签"""
        audio_path, label = self.samples[idx]

        # 加载音频（返回形状：[通道数, 采样点数]，原始采样率）
        waveform, original_sample_rate = torchaudio.load(audio_path)

        # 重采样到目标采样率
        if original_sample_rate != self.target_sample_rate:
            if original_sample_rate not in self.resamplers:
                # 缓存重采样器（按原始采样率区分）
                self.resamplers[original_sample_rate] = T.Resample(
                    orig_freq=original_sample_rate,
                    new_freq=self.target_sample_rate
                ).to(waveform.device)
            waveform = self.resamplers[original_sample_rate](waveform)

        return waveform, self.target_sample_rate, label


# -------------------------- 2. 模型类定义 --------------------------
class DepressionClassifier(nn.Module):
    """抑郁分类模型（CLAP音频编码器 + 分类头）"""

    def __init__(self, clap_model_name="laion/clap-htsat-unfused", num_classes=2):
        super().__init__()
        # 加载预训练CLAP模型（使用官方transformers接口）
        self.clap = ClapModel.from_pretrained(clap_model_name)
        # 冻结CLAP参数（仅训练分类头）
        for param in self.clap.parameters():
            param.requires_grad = False
        # 分类头（输入维度=CLAP的特征维度）
        self.classifier = nn.Linear(self.clap.config.projection_dim, num_classes)

    def forward(self, input_features):
        """
        处理输入特征并输出分类结果：
        1. input_features：ClapProcessor输出的4维张量 [B, 1, T, F]
        2. 转换为3维张量 [B, 1, T*F]（适配CLAP的audio_model）
        3. 提取特征并分类
        """
        # 转换维度：4维→3维（展平时间和频率维度）
        batch_size, channels, time_steps, freq_bins = input_features.shape
        audio_sequence = input_features.reshape(batch_size, channels, time_steps * freq_bins)

        # 提取CLAP音频特征（使用官方audio_model接口）
        audio_outputs = self.clap.audio_model(audio_sequence)
        # 取最后一层隐藏状态的平均作为特征
        audio_features = audio_outputs.last_hidden_state.mean(dim=1)

        # 分类头输出
        logits = self.classifier(audio_features)
        return logits


# -------------------------- 3. 训练配置与逻辑 --------------------------
if __name__ == "__main__":
    # --- 配置参数（按最初路径调整） ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "laion/clap-htsat-unfused"  # 预训练CLAP模型
    DATA_ROOT = "D:/PythonProject/audio-finetune/EATD_program/data"  # 你的数据根目录
    EPOCHS = 15  # 训练轮次
    BATCH_SIZE = 8  # 批次大小（显存不足可减小）
    LEARNING_RATE = 2e-5  # 学习率
    VALIDATION_SPLIT = 0.487  # 验证集比例
    RANDOM_STATE = 42  # 随机种子（保证结果可复现）

    # 音频参数（与CLAP预训练一致）
    TARGET_SAMPLE_RATE = 48000
    MAX_AUDIO_SECONDS = 10
    MAX_SAMPLES = TARGET_SAMPLE_RATE * MAX_AUDIO_SECONDS  # 480000（10秒音频的采样点数）


    # --- 数据处理函数 ---
    def collate_fn(batch):
        """统一批次内音频长度，处理为模型可接受的格式"""
        processed_waveforms = []
        labels = []
        for waveform, sample_rate, label in batch:
            # 转换为numpy数组，确保单通道（1维）
            waveform_np = waveform.numpy()
            if waveform_np.ndim > 1:
                waveform_np = waveform_np[0, :]  # 取第一通道

            # 统一长度（长裁剪，短补零）
            if len(waveform_np) > MAX_SAMPLES:
                waveform_np = waveform_np[:MAX_SAMPLES]
            else:
                pad_width = MAX_SAMPLES - len(waveform_np)
                waveform_np = np.pad(waveform_np, (0, pad_width), mode='constant')

            processed_waveforms.append(waveform_np)
            labels.append(label)

        # 提取采样率（批次内一致）
        sampling_rate = batch[0][1] if batch else TARGET_SAMPLE_RATE
        return processed_waveforms, sampling_rate, torch.tensor(labels)


    # --- 训练与评估函数 ---
    def train_one_epoch(model, data_loader, loss_fn, optimizer, scheduler, device, processor):
        """训练一轮并返回平均损失"""
        model.train()  # 训练模式
        total_loss = 0.0

        for waveforms, sample_rate, labels in tqdm(data_loader, desc="Training"):
            # 数据移至设备
            labels = labels.to(device)

            # 用CLAP处理器转换音频为特征（4维张量 [B, 1, T, F]）
            inputs = processor(
                audios=waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(device)

            # 模型预测
            predictions = model(inputs['input_features'])
            loss = loss_fn(predictions, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)


    def evaluate(model, data_loader, loss_fn, processor, device):
        """验证并返回损失、准确率、F1分数"""
        model.eval()  # 验证模式
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():  # 禁用梯度计算
            for waveforms, sample_rate, labels in tqdm(data_loader, desc="Evaluating"):
                labels = labels.to(device)

                # 特征转换
                inputs = processor(
                    audios=waveforms,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).to(device)

                # 预测
                predictions = model(inputs['input_features'])
                loss = loss_fn(predictions, labels)

                # 累计指标
                total_loss += loss.item()
                preds = torch.argmax(predictions, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return avg_loss, accuracy, f1


    # --- 主训练流程 ---
    print(f"使用设备：{DEVICE}")

    # 1. 划分训练/验证集（按志愿者ID）
    all_volunteer_ids = [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]
    train_ids, val_ids = train_test_split(
        all_volunteer_ids,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE
    )

    # 2. 初始化数据集
    train_dataset = EATDCorpusDataset(
        data_root=DATA_ROOT,
        volunteer_ids=train_ids,
        target_sample_rate=TARGET_SAMPLE_RATE
    )
    val_dataset = EATDCorpusDataset(
        data_root=DATA_ROOT,
        volunteer_ids=val_ids,
        target_sample_rate=TARGET_SAMPLE_RATE
    )

    # 3. 计算类别权重（解决样本不平衡）
    train_labels = [sample[1] for sample in train_dataset.samples]
    label_counts = Counter(train_labels)
    print(f"训练集类别分布：{label_counts}")

    # 计算权重（少数类权重 = 多数类数量 / 少数类数量）
    if label_counts[1] > 0:
        weight_for_class_1 = label_counts[0] / label_counts[1]
    else:
        weight_for_class_1 = 1.0
    class_weights = torch.tensor([1.0, weight_for_class_1], dtype=torch.float32).to(DEVICE)
    print(f"类别权重：{class_weights}")

    # 4. 初始化数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows系统建议设为0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 5. 初始化处理器和模型
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = DepressionClassifier(
        clap_model_name=MODEL_ID,
        num_classes=2  # 二分类（抑郁/非抑郁）
    ).to(DEVICE)

    # 6. 配置优化器和调度器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数数量：{sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=0.01  # 权重衰减防过拟合
    )

    # 学习率调度器（带10%预热）
    num_training_steps = EPOCHS * len(train_loader)
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"总训练步数：{num_training_steps}，预热步数：{num_warmup_steps}")

    # 7. 训练循环
    for epoch in range(EPOCHS):
        print(f"\n--- 第 {epoch + 1}/{EPOCHS} 轮 ---")
        train_loss = train_one_epoch(model, train_loader, nn.CrossEntropyLoss(weight=class_weights),
                                     optimizer, scheduler, DEVICE, processor)
        print(f"训练损失：{train_loss:.4f}")

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, nn.CrossEntropyLoss(weight=class_weights),
                                             processor, DEVICE)
        print(f"验证损失：{val_loss:.4f}，准确率：{val_acc:.4f}，F1分数：{val_f1:.4f}")

    print("\n训练完成！")