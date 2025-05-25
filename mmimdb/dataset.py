import copy
import numpy as np
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from constants import ROOT
import os
import json
import torch
from torchvision import transforms
from img_caption import ImageCaptionDataModule

class MMIMDBDataModule(ImageCaptionDataModule):
    """
    Data module for MM-IMBD vision-language dataset [1] including
    movies description (text) + poster (image).
    The downstream task is to predict the movie genre.

    [1] Gated Multimodal Units for Information Fusion, John Arevalo et al., ICLR-Workshop 2017
    """

    def __init__(self, model: str,
                 tokenizer=None,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 sample_format=None,
                 class_idx=None,
                 use_sup=False
                #  img_augment: Optional[str] = None
                 ):

        """
        :param model: {'Sup', 'SimCLR', 'CLIP', 'SLIP', 'BLIP2, 'CoMM'}
            The model defines the augmentations to apply to the data.
        :param tokenizer: Which tokenizer use for encoding text with integers
        :param batch_size: Batch size to pass to Dataloaders
        :param num_workers: Number of workers to pass to Dataloaders
        :param img_augment: What specific image augmentation to perform for SSL
        """

        super().__init__("mmimdb", model, tokenizer, batch_size, num_workers)
        self.sample_format = sample_format
        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.class_idx =class_idx
        self.use_sup = use_sup
        self.setup("test")

    def setup(self, stage: str):
        self.val_dataset = None
        root = ROOT
        if self.sample_format is not None:
            train_sampled_ids = self.sample_format.format(split="train")
            val_sampled_ids = self.sample_format.format(split="dev")
            test_sampled_ids = self.sample_format.format(split="test")
            if (self.model == "CLIP" or self.model == "CLIPSup") and not self.use_sup:
                self.train_dataset = MMIMDBDatasetSupSampled(root, train_sampled_ids, "train", 
                                                             self.img_transform, self.tokenizer, self.class_idx)
                self.val_dataset = MMIMDBDatasetSupSampled(root, val_sampled_ids, "dev", 
                                                           self.img_transform, self.tokenizer, self.class_idx)
                self.test_dataset = MMIMDBDatasetSupSampled(root, test_sampled_ids, "test", 
                                                            self.img_transform, self.tokenizer, self.class_idx)
            else:
                self.train_dataset = MMIMDBDatasetSup(root, "train", self.test_transform, self.tokenizer,
                                                      train_sampled_ids, self.class_idx)
                self.val_dataset = MMIMDBDatasetSup(root, "dev", self.test_transform, self.tokenizer,
                                                    val_sampled_ids, self.class_idx)
                self.test_dataset = MMIMDBDatasetSup(root, "test", self.test_transform, self.tokenizer,
                                                     test_sampled_ids, self.class_idx)
        else:
            if self.model == 'Sup':
                self.train_dataset = MMIMDBDatasetSup(root, "train", self.test_transform, self.tokenizer)
                self.val_dataset = MMIMDBDatasetSup(root, "dev", self.test_transform, self.tokenizer)
                self.test_dataset = MMIMDBDatasetSup(root, "test", self.test_transform, self.tokenizer)
            elif self.model == "CLIP":
                self.train_dataset = MMIMDBDatasetSup(root, "train", self.img_transform, self.tokenizer)
                self.val_dataset = MMIMDBDatasetSup(root, "dev", self.img_transform, self.tokenizer)
                self.test_dataset = MMIMDBDatasetSup(root, "test", self.img_transform, self.tokenizer)
            elif self.model == 'SupervisedClassifier':
                self.train_dataset = MMIMDBDatasetSup(root, "train", self.img_transform, self.tokenizer)
                self.val_dataset = MMIMDBDatasetSup(root, "dev", self.test_transform, self.tokenizer)
                self.test_dataset = MMIMDBDatasetSup(root, "test", self.test_transform, self.tokenizer)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True, drop_last=False)


class MMIMDBDatasetBase(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train", sampled_ids=None):
        """
        :param root: /path/to/mmimdb
        :param metadata: /path/to/mmimdb/split/ where `split.json` is located
        :param split: "train", "dev" (i.e. validation) or "test"
        """
        self.root = root
        self.split = split
        self.samples = []

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        metadata = os.path.join(root, "split.json")
        if sampled_ids is not None:
            self.ids = np.load(sampled_ids)
        else:
            with open(metadata) as f:
                self.ids = json.load(f)[self.split]
        for img_id in self.ids:
            sample_path = os.path.join(self.root, 'dataset', f'{img_id}.json')
            with open(sample_path) as f:
                meta = json.load(f)
                plot = meta['plot']
                genres = meta['genres']
            self.samples.append((img_id, plot, genres))
    
    @staticmethod
    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def get_caption(self, i):
        # if there are multiple captions, get the first one
        return self.samples[i][1][0]

    def get_raw_item(self, i):
        index, captions, genres = self.samples[i]
        path = os.path.join(self.root, 'dataset', '{}.jpeg'.format(index))
        img = self.pil_loader(path)
        caption = captions[0]

        return img, caption, genres

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

genres_ = [
    "drama", "comedy", "romance", "thriller", "crime", "action", "adventure",
    "horror", "documentary", "mystery", "sci-fi", "music", "fantasy", "family",
    "biography", "war", "history", "animation", "musical", "western", "sport",
    "short", "film-noir"
]
NUM_GENRES = len(genres_)

class MMIMDBDatasetSup(MMIMDBDatasetBase):
    def __init__(self, root, split: str = "train", transform=None, tokenizer=None, 
                 sampled_ids=None, class_idx=None):
        self.root = root
        self.split = split
        self.samples = []
        self.transform=transform
        self.tokenizer = tokenizer
        self.class_idx = class_idx
        if self.class_idx is None:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit([genres_])
        else:
            self.task_genre = genres_[self.class_idx]

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        metadata = os.path.join(root, "split.json")
        self.sampled_ids = None
        if sampled_ids is not None:
            self.ids = np.load(sampled_ids)
        else:
            with open(metadata) as f:
                self.ids = json.load(f)[self.split]
        for img_id in self.ids:
            sample_path = os.path.join(self.root, 'dataset', f'{img_id}.json')
            with open(sample_path) as f:
                meta = json.load(f)
                plot = meta['plot']
                genres = meta['genres']
            if self.class_idx is None:
                label = genres
            else:
                genres = [genre.lower() for genre in genres]
                label = 1 if self.task_genre in genres else 0
            self.samples.append((img_id, plot, label))

    def get_label(self, label):
        if self.class_idx is None:
            # one-hot encoding of genres
            genres = label
            genres = [genre.lower() for genre in genres]
            genres = self.mlb.transform([genres])[0]
            return genres
        else:
            return label

    def __getitem__(self, i):
        img, caption, label = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.test_transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return (img, caption), self.get_label(label)

    def get_captions(self):
        return [self.get_caption(i) for i in range(len(self.samples))]
    

class MMIMDBDatasetSupSampled(MMIMDBDatasetBase):
    def __init__(self, root, sampled_ids, split: str = "train", transform=None, tokenizer=None, 
                 class_idx=None):
        self.root = root
        self.split = split
        self.samples = []
        self.transform=transform
        self.tokenizer = tokenizer
        self.class_idx = class_idx
        if self.class_idx is None:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit([genres_])
        else:
            self.task_genre = genres_[self.class_idx]

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.sampled_ids = np.load(sampled_ids)
        metadata = os.path.join(root, "split.json")
        with open(metadata) as f:
            self.ids = json.load(f)[self.split]
        for img_id in self.ids:
            sample_path = os.path.join(self.root, 'dataset', f'{img_id}.json')
            with open(sample_path) as f:
                meta = json.load(f)
                plot = meta['plot']
                genres = meta['genres']
            if img_id not in self.sampled_ids or self.class_idx is None:
                label = -1
            else:
                genres = [genre.lower() for genre in genres]
                label = 1 if self.task_genre in genres else 0
            self.samples.append((img_id, plot, label))

    def __getitem__(self, i):
        img, caption, label = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.test_transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return (img, caption), label
    

class MMIMDBFeaturesDatasetSup(torch.utils.data.Dataset):
    def __init__(self, feat_path, root, split="train", layer_idx=-1, sampled_ids=None):
        self.root = root
        self.features = torch.load(feat_path)["feats"][:, layer_idx, :]
        self.features = self.features.float()
        self.split = split
        assert sampled_ids is not None, "sampled_ids must be provided"
        if sampled_ids is not None:
            self.ids = np.load(sampled_ids)
        self.labels = []
        for img_id in self.ids:
            sample_path = os.path.join(self.root, 'dataset', f'{img_id}.json')
            with open(sample_path) as f:
                meta = json.load(f)
                genres = meta['genres']
                self.labels.append(genres)
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([genres_])
    
    def get_labels(self):
        all_labels = [self.__getitem__(i)[1] for i in range(len(self.labels))]
        return np.stack(all_labels)

    def __getitem__(self, i):
        features, genres = self.features[i], self.labels[i]
        assert len(genres) > 0
        # one-hot encoding of genres
        genres = [genre.lower() for genre in genres]
        genres = self.mlb.transform([genres])[0]
        return features, genres
    
    def __len__(self):
        return len(self.features)
    

def sample_dataset(dataset, save_path, num_samples=1024, seed=0):
    rng = np.random.default_rng(seed)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    dataset_ids = copy.deepcopy(dataset.ids)
    rng.shuffle(dataset_ids)
    sampled_ids = []
    dataset_idx = 0
    save_idx = 0
    while save_idx < num_samples:
        curr_id = dataset_ids[dataset_idx]
        sample_path = os.path.join(ROOT, 'dataset', f'{curr_id}.json')
        with open(sample_path) as f:
            meta = json.load(f)
        captions = meta['plot']
        # Check that the caption is not empty
        if len(captions[0].split(' ')) > 1:
            sampled_ids.append(curr_id)
            save_idx += 1
            break

        dataset_idx += 1
    print(save_path)
    np.save(save_path, sampled_ids)

if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        sampled_save_path = os.path.join(ROOT, f"{split}/sampled_ids")
        metadata = ROOT
        dataset = MMIMDBDatasetSup(ROOT, split=split)
        sample_dataset(dataset, sampled_save_path)