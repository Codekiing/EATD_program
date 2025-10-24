import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


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