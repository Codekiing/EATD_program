"""
prepare_eatd_from_folders.py

假设原始结构:
root_source/
├── session_0001/
│   ├── negative.wav
│   ├── positive.wav
│   └── neutral.wav
├── session_0002/
│   ├── negative.wav
│   ├── positive.wav
│   └── neutral.wav
...

目标输出结构 (target_root):
target_root/
├── train/
│   ├── negative/
│   │   ├── negative_session_0001.wav
│   │   └── ...
│   ├── positive/
│   └── neutral/
├── test/
│   ├── negative/
│   ├── positive/
│   └── neutral/

用法:
python prepare_eatd_from_folders.py
"""

import os
import shutil
import random
from tqdm import tqdm

# ---------- 配置 ----------
SOURCE_ROOT = r"D:\PythonProject\audio-finetune\EATD_program\EATD-Corpus"  # 原始200个子文件夹所在目录
TARGET_ROOT = r"D:\PythonProject\audio-finetune\EATD_program\data"  # 输出根目录（会创建 train/test/...）
TEST_RATIO = 0.2   # 测试集比例
SEED = 42
COPY_INSTEAD_OF_MOVE = True  # True: copy文件; False: move文件（会从源目录移除）
OVERWRITE = False  # True: 若目标已存在则覆盖
# -------------------------

random.seed(SEED)

LABEL_FILES = {
    "negative": "negative.wav",
    "positive": "positive.wav",
    "neutral": "neutral.wav"
}


def sanitize_name(s: str) -> str:
    """把目录名转换成文件名可用的安全字符串"""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def collect_and_rename(source_root, tmp_out_dir):
    """
    遍历 source_root 下的每个子文件夹:
    - 如果找到 negative/positive/neutral wav，则拷贝(或移动)到 tmp_out_dir/<label>/ 并重命名为 <label>_<foldername>.wav
    返回值: dict(label -> list of target file paths)
    """
    os.makedirs(tmp_out_dir, exist_ok=True)
    for label in LABEL_FILES:
        os.makedirs(os.path.join(tmp_out_dir, label), exist_ok=True)

    label_to_files = {label: [] for label in LABEL_FILES}

    subdirs = sorted([d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))])
    if not subdirs:
        raise RuntimeError(f"在 {source_root} 没有发现子文件夹，请确认路径正确。")

    print(f"发现 {len(subdirs)} 个子文件夹，开始收集并重命名音频...")
    for folder in tqdm(subdirs):
        folder_path = os.path.join(source_root, folder)
        safe_folder = sanitize_name(folder)
        for label, filename in LABEL_FILES.items():
            src_file = os.path.join(folder_path, filename)
            if not os.path.isfile(src_file):
                # 如果缺失，打印警告并跳过
                print(f"⚠️ 警告: 在 '{folder}' 下未找到 {filename}，跳过该文件 (label={label}).")
                continue

            new_fname = f"{label}_{safe_folder}.wav"
            dst_path = os.path.join(tmp_out_dir, label, new_fname)

            if os.path.exists(dst_path):
                if OVERWRITE:
                    os.remove(dst_path)
                else:
                    # 若存在且不覆盖，则为避免冲突添加后缀
                    base, ext = os.path.splitext(new_fname)
                    idx = 1
                    while os.path.exists(os.path.join(tmp_out_dir, label, f"{base}_{idx}{ext}")):
                        idx += 1
                    new_fname = f"{base}_{idx}{ext}"
                    dst_path = os.path.join(tmp_out_dir, label, new_fname)

            if COPY_INSTEAD_OF_MOVE:
                shutil.copy2(src_file, dst_path)
            else:
                shutil.move(src_file, dst_path)

            label_to_files[label].append(dst_path)

    return label_to_files


def split_and_create_dataset(tmp_out_dir, target_root, test_ratio=0.2):
    """
    对 tmp_out_dir/<label> 下的文件按 test_ratio 划分 train/test，
    并把文件(复制或移动)到 target_root/train/<label>/ 和 target_root/test/<label>/
    """
    train_dir = os.path.join(target_root, "train")
    test_dir = os.path.join(target_root, "test")

    for base in [train_dir, test_dir]:
        os.makedirs(base, exist_ok=True)

    for label in LABEL_FILES:
        src_label_dir = os.path.join(tmp_out_dir, label)
        files = sorted([os.path.join(src_label_dir, f) for f in os.listdir(src_label_dir) if f.lower().endswith(".wav")])
        if not files:
            print(f"⚠️ 警告: 在临时目录未找到类别 '{label}' 的任何文件，跳过该类别。")
            continue

        random.shuffle(files)
        split_idx = int(len(files) * (1 - test_ratio))
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        out_train_label_dir = os.path.join(train_dir, label)
        out_test_label_dir = os.path.join(test_dir, label)
        os.makedirs(out_train_label_dir, exist_ok=True)
        os.makedirs(out_test_label_dir, exist_ok=True)

        print(f"\n类别 '{label}' 共 {len(files)} 个样本，划分为 {len(train_files)} train / {len(test_files)} test")

        # 移动/复制到目标目录
        for src in train_files:
            dst = os.path.join(out_train_label_dir, os.path.basename(src))
            if os.path.exists(dst):
                if OVERWRITE:
                    os.remove(dst)
                else:
                    # 避免冲突加入索引
                    base, ext = os.path.splitext(os.path.basename(dst))
                    idx = 1
                    while os.path.exists(os.path.join(out_train_label_dir, f"{base}_{idx}{ext}")):
                        idx += 1
                    dst = os.path.join(out_train_label_dir, f"{base}_{idx}{ext}")

            if COPY_INSTEAD_OF_MOVE:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

        for src in test_files:
            dst = os.path.join(out_test_label_dir, os.path.basename(src))
            if os.path.exists(dst):
                if OVERWRITE:
                    os.remove(dst)
                else:
                    base, ext = os.path.splitext(os.path.basename(dst))
                    idx = 1
                    while os.path.exists(os.path.join(out_test_label_dir, f"{base}_{idx}{ext}")):
                        idx += 1
                    dst = os.path.join(out_test_label_dir, f"{base}_{idx}{ext}")

            if COPY_INSTEAD_OF_MOVE:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)


def main():
    print("开始预处理 EATD 样本文件夹 ...")
    tmp_out_dir = os.path.join(TARGET_ROOT, "dataset-EATD")
    # 清理 tmp 目录（如果存在）
    if os.path.exists(tmp_out_dir):
        print(f"存在旧临时目录 {tmp_out_dir}，将被清空（如果你想保留，请先修改脚本）")
        shutil.rmtree(tmp_out_dir)
    os.makedirs(tmp_out_dir, exist_ok=True)

    # 第一步：收集并按 label 存放到临时目录（并重命名以避免冲突）
    label_to_files = collect_and_rename(SOURCE_ROOT, tmp_out_dir)

    # 第二步：按 label 划分 train/test 并放到目标目录
    split_and_create_dataset(tmp_out_dir, TARGET_ROOT, test_ratio=TEST_RATIO)

    # 可选：删除临时目录（如果你希望保留可以注释掉）
    print(f"\n处理完成。临时目录（{tmp_out_dir}）将被删除。")
    shutil.rmtree(tmp_out_dir)

    print(f"\n最终数据组织完成：")
    print(f"  训练集: {os.path.join(TARGET_ROOT, 'train')}")
    print(f"  测试集: {os.path.join(TARGET_ROOT, 'test')}")
    print("\n请在训练脚本中使用如下路径：")
    print(f"DATA_DIR = r\"{TARGET_ROOT}\"")
    print("train_dataset = EATDAudioDataset(DATA_DIR, split='train', spec_size=spec_size)")
    print("test_dataset  = EATDAudioDataset(DATA_DIR, split='test', spec_size=spec_size)")
    print("\n若需要我把脚本改为直接生成 .npy 梅尔谱或额外预处理（重采样/截取固定长度/创建 spectrogram），告诉我，我可以直接把那部分也加上。")


if __name__ == "__main__":
    main()
