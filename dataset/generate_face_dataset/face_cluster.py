import glob
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms import ToTensor
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
import os
import json
import shutil
from os.path import join
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(150, 150),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
root_path = '/root/visualize'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
shutil.rmtree(root_path, ignore_errors=True)

name2path = json.load(open('/root/data/wzy/datalocker/video_name_to_path.json'))
all_real_images = glob.glob('/root/face_to_be_clustered/*.png')

part2images = {}
for real_image in tqdm(all_real_images):
    image_name = real_image.split('/')[-1]
    image_name = image_name.split('_')[0]
    image_path = name2path[image_name + '.mp4']
    part2images.setdefault(image_path.split('/')[-2], []).append(real_image)

tf_img = lambda i: transform(i).unsqueeze(0)
embeddings = lambda i: resnet(i)
all_real_embeddings = []
missing_videos = []

for face_path in tqdm(part2images['dfdc_train_part_38']):
    try:
        t = tf_img(Image.open(face_path)).to(device)
        e = embeddings(t).squeeze().cpu().tolist()
        all_real_embeddings.append(e)
    except:
        missing_videos.append(face_path)

labels = AgglomerativeClustering(n_clusters=None, affinity="cosine", linkage='complete',
                                 distance_threshold=1).fit_predict(all_real_embeddings)

for img_path, label in zip(part2images['dfdc_train_part_38'], labels):
    save_path = join(root_path, str(label))
    os.makedirs(save_path, exist_ok=True)
    image_name = img_path.split('/')[-1]
    shutil.copy(img_path, join(save_path, image_name))

json.dump(missing_videos, open('missing_video.json', 'w'))
