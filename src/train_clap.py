import os
import json
import torch
import warnings
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from laion_clap.clap_module.model import CLAP
from dataset import EATDAudioDataset  # 导入修改后的数据集


# 屏蔽torch.meshgrid警告
warnings.filterwarnings("ignore",
                        message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")


class AudioClassifier(torch.nn.Module):
    """音频情感分类器（CLAP音频编码器 + 分类头）"""

    def __init__(self, clap_model, embed_dim, num_classes=3):
        super().__init__()
        self.clap_audio_encoder = clap_model.audio_infer  # 音频推理接口
        self.fc = torch.nn.Linear(embed_dim, num_classes)  # 分类头

    # def forward(self, audio):
    #     """
    #     接受 audio 两种可能形态:
    #       - [B, 1, L]
    #       - [B, L]
    #     对 batch 中每个样本单独调用 clap_model.audio_infer（它要求 1D 时域音频）。
    #     将 output_dict 的第一个 value 取出作为 embedding（若是多帧则取 mean）。
    #     最后返回 logits: [B, num_classes]
    #     """
    #
    #     # 规范成 [B, L]
    #     if audio.ndim == 3 and audio.shape[1] == 1:
    #         audio = audio.squeeze(1)  # [B, L]
    #     elif audio.ndim == 4 and audio.shape[1] == 1 and audio.shape[3] == 1:
    #         # 如果你之前传的是 [B,1,L,1] 的情况
    #         audio = audio.squeeze(1).squeeze(-1)  # -> [B, L]
    #
    #     B = audio.shape[0]
    #     device = audio.device
    #
    #     embeddings = []
    #     # 对每个样本逐条调用 audio_infer（audio_infer 要求单条 1D tensor）
    #     for i in range(B):
    #         single = audio[i]  # shape [L], 1D tensor
    #
    #         # 如果 single 在 CPU 上或 GPU 上， audio_infer 接受 device 参数，传 device 保证一致
    #         out_dict = self.clap_audio_encoder(single, device=device)
    #
    #         # out_dict 是 dict，key 名称可能不同，取第一个 value（最通用）
    #         if not out_dict:
    #             raise RuntimeError("clap.audio_infer 返回了空字典，请检查模型或输入。")
    #         first_val = next(iter(out_dict.values()))
    #
    #         # first_val 可能是 1D (embed,) 或 2D (n_frames, embed)
    #         if first_val.ndim == 1:
    #             emb = first_val
    #         else:
    #             # 如果是多帧 (n_frames, embed)，将其按时间维度做 pool（mean pool）
    #             emb = first_val.mean(dim=0)
    #
    #         embeddings.append(emb)
    #
    #     # stack -> [B, embed_dim]
    #     embeddings = torch.stack(embeddings, dim=0).to(device)
    #
    #     logits = self.fc(embeddings)
    #     return logits

    def forward(self, audio):
        # 关键：调整输入维度以匹配HTSAT要求（[B, L, C]）
        # 输入audio形状为[B, 1, L, 1]，需移除多余的维度
        audio = audio.squeeze(1).squeeze(-1)  # 移除中间的1维度 → [B, L, 1]（符合[B, L, C]，C=1）
        # 提取音频特征
        audio_features = self.clap_audio_encoder(audio)
        logits = self.fc(audio_features)
        return logits


if __name__ == '__main__':
    # -------------------------- 配置参数 --------------------------
    DATA_DIR = "D:/PythonProject/audio-finetune/EATD_program/data"
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-5
    SAVE_DIR = "D:/PythonProject/audio-finetune/EATD_program/checkpoints"
    CONFIG_PATH = "D:/PythonProject/audio-finetune/EATD_program/CLAP/src/laion_clap/clap_module/model_configs/HTSAT-base.json"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # -------------------------- 加载配置 --------------------------
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    config["model_name"] = "htsat-base"
    audio_cfg = config["audio_cfg"]
    text_cfg = config["text_cfg"]
    text_cfg["model_type"] = "roberta"

    # 从配置中提取频谱图尺寸（与dataset.py匹配）
    spec_size = audio_cfg.get("spec_size", 256)  # 从HTSAT配置获取，默认256
    print(f"HTSAT频谱图尺寸：{spec_size}x{spec_size}，序列长度L={spec_size*spec_size}")

    # -------------------------- 初始化CLAP模型并冻结 --------------------------
    clap_model = CLAP(
        embed_dim=config["embed_dim"],
        audio_cfg=audio_cfg,
        text_cfg=text_cfg,
        quick_gelu=config.get("quick_gelu", False),
        enable_fusion=False
    ).to(device)

    clap_model.eval()  # 强制CLAP处于eval模式（满足audio_infer断言）
    for param in clap_model.parameters():
        param.requires_grad = False  # 冻结CLAP参数

    # -------------------------- 初始化分类模型 --------------------------
    model = AudioClassifier(
        clap_model=clap_model,
        embed_dim=config["embed_dim"],
        num_classes=3
    ).to(device)

    # -------------------------- 数据加载（传递spec_size参数） --------------------------
    # 关键：将HTSAT配置中的spec_size传给数据集，确保频谱图尺寸匹配
    train_dataset = EATDAudioDataset(
        DATA_DIR,
        split="train",
        spec_size=spec_size  # 与模型配置一致
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    test_dataset = EATDAudioDataset(
        DATA_DIR,
        split="test",
        spec_size=spec_size  # 与模型配置一致
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    # 打印数据集信息
    print(f"\n训练集文件数：{len(train_dataset)}，测试集文件数：{len(test_dataset)}")
    if len(train_dataset) > 0:
        print("前3个训练标签：", train_dataset.labels[:3])

    # -------------------------- 训练配置 --------------------------
    optimizer = Adam(model.fc.parameters(), lr=LEARNING_RATE)  # 仅优化分类头
    criterion = CrossEntropyLoss()

    # -------------------------- 训练循环 --------------------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0.0

        for audio, labels in train_loader:
            audio = audio.to(device)
            # 打印调整前的形状（验证数据集输出是否正确）
            print(f"批次原始形状：{audio.shape}（预期：({BATCH_SIZE}, 1, {spec_size*spec_size}, 1)）")
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio)  # 经过forward处理后适配HTSAT
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * audio.size(0)
            _, preds = torch.max(outputs, 1)
            # train_correct += torch.sum(preds == labels.data)
            train_correct += (preds == labels.data).sum().item()  # 改这里

        train_avg_loss = train_loss / len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0.0  # 改成浮点型

        with torch.no_grad():
            for audio, labels in test_loader:
                audio = audio.to(device)
                labels = labels.to(device)
                outputs = model(audio)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * audio.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels.data).sum().item()  # 改这里

        val_avg_loss = val_loss / len(test_dataset)
        val_acc = val_correct / len(test_dataset)  # 不再需要 .double()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"训练：损失={train_avg_loss:.4f}，准确率={train_acc:.4f}")
        print(f"验证：损失={val_avg_loss:.4f}，准确率={val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"clap_classifier_epoch_{epoch + 1}.pth"))

    print("\n训练完成！模型已保存至：", SAVE_DIR)