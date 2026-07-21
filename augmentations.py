import random
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms

class RandomJPEGCompression:
    def __init__(self, quality=(30, 100)):
        self.quality = quality

    def __call__(self, img):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf)

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        noisy = np.clip(arr + np.random.normal(self.mean, self.std, arr.shape), 0, 1) * 255
        return Image.fromarray(noisy.astype(np.uint8))

class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.5, 2.0)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img)
        h, w = arr.shape[:2]
        area = h * w
        for _ in range(10):
            target = random.uniform(*self.scale) * area
            ar = random.uniform(*self.ratio)
            eh = int(round(np.sqrt(target * ar)))
            ew = int(round(np.sqrt(target / ar)))
            if eh < h and ew < w:
                top = random.randint(0, h - eh)
                left = random.randint(0, w - ew)
                arr[top:top+eh, left:left+ew] = 0
                break
        return Image.fromarray(arr)

def get_transform(name):
    if name == 'raaug':
        def raaug(img):
            choice = random.choice(['none', 'erase', 'randcrop'])
            if choice == 'none':
                img = transforms.Resize((224, 224))(img)
            elif choice == 'erase':
                img = transforms.Resize((224, 224))(img)
                img = RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.5, 2.0))(img)
            else:
                img = transforms.RandomResizedCrop(224, scale=(1/1.3, 1.0), ratio=(0.9, 1.1))(img)
            return transforms.ToTensor()(img)
        return transforms.Lambda(raaug)

    elif name == 'dfdcselim':
        return transforms.Compose([
            transforms.RandomApply([RandomJPEGCompression(quality=(30, 100))], p=0.5),
            transforms.RandomApply([AddGaussianNoise(mean=0, std=0.1)], p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),
            transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05))], p=0.3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    else:
        raise ValueError(f"Unknown transform: {name}")
