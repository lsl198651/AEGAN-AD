import os
import numpy as np
import librosa
import logging
import datetime
import random
import torch
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi
from tqdm import tqdm

from maps import CLASS_ATTRI_ALL, CLASS_SEC_ATTRI, ATTRI_CODE


def get_wav_data(flag: str):
    npy_path_padded = r"D:\Shilong\murmur\01_dataset\04_newDataset\npyFile_padded\normalized\list_npy_files"
# 读取npy文件
    if flag == 'train':
        absent_train_features = np.load(
            npy_path_padded + r"\absent_train_features_norm.npy", allow_pickle=True
        )
        absent_train_label = np.load(
            npy_path_padded + r"\absent_train_label_norm.npy", allow_pickle=True
        )
        # absent_train_features = torch.tensor(absent_train_features)
        # all_mel_features = []
        # for index in tqdm(range(len(absent_train_features))):
        #     mel_features = ta_kaldi.fbank(absent_train_features[index:], num_mel_bins=128, sample_frequency=16000,
        #                                   frame_length=25, frame_shift=10, window_type='hamming')
        #     all_mel_features.append(mel_features)
        # all_fbank_features = torch.stack(all_mel_features, dim=0)
        return absent_train_features, absent_train_label
    elif flag == 'test':
        absent_test_features = np.load(
            npy_path_padded + r"\absent_test_features_norm.npy", allow_pickle=True
        )
        present_test_features = np.load(
            npy_path_padded + r"\present_test_features_norm.npy", allow_pickle=True
        )
        absent_test_label = np.load(
            npy_path_padded + r"\absent_test_label_norm.npy", allow_pickle=True
        )
        present_test_label = np.load(
            npy_path_padded + r"\present_test_label_norm.npy", allow_pickle=True
        )
        test_feature = np.vstack((absent_test_features, present_test_features))
        test_label = np.hstack((absent_test_label, present_test_label))
        return test_feature, test_label
    else:
        raise ValueError("flag must be 'train' or 'test'")


def generate_spec(clip_addr,  top_dir=None):
    all_clip_spec = None
    top_dir = r'D:\Shilong\murmur\01_dataset\04_newDataset\trainset'

    for set_type in clip_addr.keys():  # 'dev', 'eval'
        save_dir = os.path.join(top_dir, set_type)
        os.makedirs(save_dir, exist_ok=True)
        # 设置保存路径
        #  mel_bin=128, frame_hop=512,fft_num=2048
        mel_bin = 128
        raw_data_file = os.path.join(
            save_dir, 'raw_logmel_128bin_2048fft_512_1.npy')

        if not os.path.exists(raw_data_file):
            for idx in tqdm(range(len(clip_addr[set_type]))):
                wavform = torch.tensor(clip_addr[set_type][idx])
                # 读取wav文件，提取logmel谱fbank
                fbank = ta_kaldi.fbank(
                    wavform.long(), num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10, window_type='hamming')
                fbank = fbank.numpy().T  # (128,22)
                if idx == 0:
                    # 创建全0矩阵，(3000*128)*313
                    set_clip_spec = np.zeros(
                        (len(clip_addr[set_type]) * mel_bin, fbank.shape[1]), dtype=np.float32)
                set_clip_spec[idx * mel_bin:(idx + 1) * mel_bin, :] = fbank
            np.save(raw_data_file, set_clip_spec)  # 保存fbank数据
        else:
            set_clip_spec = np.load(raw_data_file)
        if all_clip_spec is None:
            all_clip_spec = set_clip_spec
        else:
            all_clip_spec = np.vstack((all_clip_spec, set_clip_spec))
            # 把所有logmel谱按照纵向排列堆叠，（3000*128）*313 = all_clip_spec
    # 求出长度为313的logmel谱，每个logmel谱的长度为3000*128
    frame_num_per_clip = all_clip_spec.shape[-1]  # 313
    max_v = np.max(all_clip_spec)
    min_v = np.min(all_clip_spec)
    print('fbank scale max: {}, min: {}'.format(max_v, min_v))

    fbank_mean = (max_v + min_v) / 2  # 均值
    scale = (max_v - min_v) / 2  # 范围
    fbank_std = np.std(all_clip_spec)
    all_clip_spec = (all_clip_spec - fbank_mean) / scale  # 去直流，归一化
    all_clip_spec = (all_clip_spec - fbank_mean) / fbank_std  # 去直流，标准化
    # reshape fbank 特征
    all_clip_spec = all_clip_spec.reshape(-1, mel_bin, frame_num_per_clip)
    return all_clip_spec


def generate_label(clip_addr, set_type, data_type):
    label = np.zeros((len(clip_addr)), dtype=int)
    # normal: 0, anomaly: 1 dev_test
    # normal: 0, dev_train
    # normal: -1, anomaly: -1, eval_test
    for idx in range(len(clip_addr)):
        # train：section_01_target_train_normal_0000_f-n_C.wav
        # test：section_01_target_test_normal_0029_f-n_C.wav
        if set_type == 'dev' and data_type == 'test':
            status_note = clip_addr[idx].split('\\')[-1].split('_')[4]
            assert status_note in ['normal', 'anomaly']
            status = 0 if status_note == 'normal' else 1
        elif data_type == 'train':
            status = 0
        else:  # for eval test
            status = -1
        label[idx] = status
    return label


def extract_attri(clip_addr, mt, eval_te_flag=False):
    # train：section_01_target_train_normal_0000_f-n_C.wav
    # test：section_01_target_test_normal_0029_f-n_C.wav
    # list of list, [sec, domain, att0, att1, att2, ...]
    # 对每个wav创建空属性列表
    all_attri = [[] for _ in clip_addr]
    attri_idx = CLASS_SEC_ATTRI[mt]
    for cid, clip in enumerate(clip_addr):
        # 提取出文件名file_name，不包含扩展名
        file_name = os.path.basename(
            clip)[:os.path.basename(clip).index('.wav')]
        segs = file_name.split('_')
        # 从文件名提取出sec和domain
        sec, domain_note = int(segs[1][1]), segs[2]
        # 属性source=0，target=1
        domain = 0 if domain_note == 'source' else 1
        # extraction of auxiliary scene labels, not used in training
        # 向all_attri中添加sec属性
        all_attri[cid].append(sec)
        if not eval_te_flag:
            # 向all_attri中添加domain属性
            all_attri[cid].append(domain)
            sec_attri = attri_idx[sec]
            for atn in CLASS_ATTRI_ALL[mt]:
                if atn not in sec_attri:
                    atv = 'none'
                else:
                    if mt == 'valve' and atn == 'v':
                        if 'v1pat' in segs and 'v2pat' in segs:
                            atv = 'v1pat_v2pat'
                        elif 'v1pat' in segs:
                            atv = 'v1pat'
                        elif 'v2pat' in segs:
                            atv = 'v2pat'
                        else:
                            atv = 'none'
                    else:
                        assert atn in segs
                        atv = segs[segs.index(atn) + 1]
                all_attri[cid].append(
                    ATTRI_CODE[mt][atn][atv])  # value to code
    return np.array(all_attri)


def weights_init(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


def get_model_pth(param):
    return os.path.join(param['model_dir'], f"{param['mt']}.pth")


def config_summary(param):
    summary = {}
    summary['feat'] = {'fft_num': param['feat']['fft_num'],
                       'mel_bin': param['feat']['mel_bin'],
                       'frame_hop': param['feat']['frame_hop'],
                       'graph_hop_f': param['feat']['graph_hop_f']}
    summary['set'] = {'dataset': param['train_set']}
    summary['net'] = {'act': param['net']['act'],
                      'normalize': param['net']['normalize'],
                      'nz': param['net']['nz'],
                      'ndf': param['net']['ndf'],
                      'ngf': param['net']['ngf']}
    summary['train'] = {'lrD': param['train']['lrD'],
                        'lrG': param['train']['lrG'],
                        'batch_size': param['train']['batch_size'],
                        'epoch': param['train']['epoch']}
    summary['wgan'] = param['train']['wgan']
    return summary


def get_logger(param):
    os.makedirs(os.path.join('./log'), exist_ok=True)
    log_name = '././log/train.log'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    # fh = logging.FileHandler(log_name, mode='a' if param['resume'] else 'w')
    sh_form = logging.Formatter('%(message)s')
    # fh_form = logging.Formatter('%(levelname)s - %(message)s')
    sh.setLevel(logging.INFO)
    # fh.setLevel(logging.DEBUG)
    sh.setFormatter(sh_form)
    # fh.setFormatter(fh_form)
    logger.addHandler(sh)
    # logger.addHandler(fh)
    logger.info('Train starts at: {}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    return logger


def get_mle_features(path):
    """按照path读取每个wav文件，
    提取logmel谱，然后保存到mle_features文件夹中保存为feature.npy供使用

    也可以直接读wav数据传入dataset
    然后在网络中加入提取logmel谱的层来回去mei谱
    """

    return mle_features
