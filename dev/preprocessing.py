from dataset import PreProcessingDataset

if __name__ == '__main__':

    path = '/data/SunYang/datasets/Algonauts_dataset'
    subj = 1
    subj_dir = 'subj' + format(subj, '02')
    dataset = PreProcessingDataset(path, subj_dir)