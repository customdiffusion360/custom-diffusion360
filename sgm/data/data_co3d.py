# code taken and modified from https://github.com/amyxlase/relpose-plus-plus/blob/b33f7d5000cf2430bfcda6466c8e89bc2dcde43f/relpose/dataset/co3d_v2.py#L346)
import gzip
import json
import os.path as osp

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageFile
from pytorch3d.implicitron.dataset.utils import (
    adjust_camera_to_bbox_crop_,
    adjust_camera_to_image_scale_,
)
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import Rotate, Translate
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CO3D_DIR = "data/training/"

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Added: normalize camera poses
def intersect_skew_line_groups(p, r, mask):
    # p, r both of shape (B, N, n_intersected_lines, 3)
    # mask of shape (B, N, n_intersected_lines)
    p_intersect, r = intersect_skew_lines_high_dim(p, r, mask=mask)
    _, p_line_intersect = _point_line_distance(
        p, r, p_intersect[..., None, :].expand_as(p)
    )
    intersect_dist_squared = ((p_line_intersect - p_intersect[..., None, :]) ** 2).sum(
        dim=-1
    )
    return p_intersect, p_line_intersect, intersect_dist_squared, r


def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    if torch.any(torch.isnan(p_intersect)):
        print(p_intersect)
        assert False
    return p_intersect, r


def _point_line_distance(p1, r1, p2):
    df = p2 - p1
    proj_vector = df - ((df * r1).sum(dim=-1, keepdim=True) * r1)
    line_pt_nearest = p2 - proj_vector
    d = (proj_vector).norm(dim=-1)
    return d, line_pt_nearest


def compute_optical_axis_intersection(cameras):
    centers = cameras.get_camera_center()
    principal_points = cameras.principal_point

    one_vec = torch.ones((len(cameras), 1))
    optical_axis = torch.cat((principal_points, one_vec), -1)

    pp = cameras.unproject_points(optical_axis, from_ndc=True, world_coordinates=True)

    pp2 = torch.zeros((pp.shape[0], 3))
    for i in range(0, pp.shape[0]):
        pp2[i] = pp[i][i]

    directions = pp2 - centers
    centers = centers.unsqueeze(0).unsqueeze(0)
    directions = directions.unsqueeze(0).unsqueeze(0)

    p_intersect, p_line_intersect, _, r = intersect_skew_line_groups(
        p=centers, r=directions, mask=None
    )

    p_intersect = p_intersect.squeeze().unsqueeze(0)
    dist = (p_intersect - centers).norm(dim=-1)

    return p_intersect, dist, p_line_intersect, pp2, r


def normalize_cameras(cameras, scale=1.0):
    """
    Normalizes cameras such that the optical axes point to the origin and the average
    distance to the origin is 1.

    Args:
        cameras (List[camera]).
    """

    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()
    new_transform = new_cameras.get_world_to_view_transform()

    p_intersect, dist, p_line_intersect, pp, r = compute_optical_axis_intersection(
        cameras
    )
    t = Translate(p_intersect)

    # scale = dist.squeeze()[0]
    scale = max(dist.squeeze())

    # Degenerate case
    if scale == 0:
        print(cameras.T)
        print(new_transform.get_matrix()[:, 3, :3])
        return -1
    assert scale != 0

    new_transform = t.compose(new_transform)
    new_cameras.R = new_transform.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform.get_matrix()[:, 3, :3] / scale
    return new_cameras, p_intersect, p_line_intersect, pp, r


def centerandalign(cameras):

    new_cameras = join_cameras_as_batch([cameras[i].clone() for i in range(len(cameras))])
    cam_trans = new_cameras.get_world_to_view_transform().inverse()
    eye_at_up_view = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32, device=cam_trans.device
    )
    eye_at_up_world = cam_trans.transform_points(eye_at_up_view).reshape(-1, 3, 3)

    eye, at, up_plus_eye = eye_at_up_world.unbind(1)
    up = up_plus_eye - eye
    up = torch.mean(up, dim=0)

    centers = [cam.get_camera_center() for cam in new_cameras]
    centers = torch.concat(centers, 0)

    n = up / np.linalg.norm(up)

    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(n, [0, 1, 0])
    s = np.linalg.norm(v)
    c = np.dot(n, [0, 1, 0])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot = torch.from_numpy(np.eye(3) + V + V @ V * (1 - c) / s**2).float()

    new_transform = new_cameras.get_world_to_view_transform()
    rot = Rotate(rot.T)

    new_transform = rot.compose(new_transform)
    new_cameras.R = new_transform.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform.get_matrix()[:, 3, :3]
    return new_cameras


def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.

    Args:
        bbox: Bounding box in xyxy format (4,).

    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = ((bbox[:2] + bbox[2:]) / 2).round().astype(int)
    extents = (bbox[2:] - bbox[:2]) / 2
    s = (max(extents) * (1 + padding)).round().astype(int)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )

    return square_bbox


class Co3dDataset(Dataset):
    def __init__(
        self,
        category,
        split="train",
        skip=2,
        img_size=1024,
        num_images=4,
        mask_images=False,
        single_id=0,
        bbox=False,
        modifier_token=None,
        addreg=False,
        drop_ratio=0.5,
        drop_txt=0.1,
        categoryname=None,
        aligncameras=False,
        repeat=100,
        addlen=False,
        onlyref=False,
    ):
        """
        Args:
            category (iterable): List of categories to use. If "all" is in the list,
                all training categories are used.
            num_images (int): Default number of images in each batch.
            normalize_cameras (bool): If True, normalizes cameras so that the
                intersection of the optical axes is placed at the origin and the norm
                of the first camera translation is 1.
            mask_images (bool): If True, masks out the background of the images.
        """
        category = sorted(category.split(','))
        self.category = category
        self.single_id = single_id
        self.addlen = addlen
        self.onlyref = onlyref
        self.categoryname = categoryname
        self.bbox = bbox
        self.modifier_token = modifier_token
        self.addreg = addreg
        self.drop_txt = drop_txt
        self.skip = skip
        if self.addreg:
            with open(f'data/regularization/{category[0]}_sp_generated/caption.txt', "r") as f:
                self.regcaptions = f.read().splitlines()
            self.reglen = len(self.regcaptions)
            self.regimpath = f'data/regularization/{category[0]}_sp_generated'

        self.low_quality_translations = []
        self.rotations = {}
        self.category_map = {}
        co3d_dir = CO3D_DIR
        for c in category:
            subset = 'fewview_dev'
            category_dir = osp.join(co3d_dir, c)
            frame_file = osp.join(category_dir, "frame_annotations.jgz")
            sequence_file = osp.join(category_dir, "sequence_annotations.jgz")
            subset_lists_file = osp.join(category_dir, f"set_lists/set_lists_{subset}.json")
            bbox_file = osp.join(category_dir, f"{c}_bbox.jgz")

            with open(subset_lists_file) as f:
                subset_lists_data = json.load(f)

            with gzip.open(sequence_file, "r") as fin:
                sequence_data = json.loads(fin.read())

            with gzip.open(bbox_file, "r") as fin:
                bbox_data = json.loads(fin.read())

            with gzip.open(frame_file, "r") as fin:
                frame_data = json.loads(fin.read())

            frame_data_processed = {}
            for f_data in frame_data:
                sequence_name = f_data["sequence_name"]
                if sequence_name not in frame_data_processed:
                    frame_data_processed[sequence_name] = {}
                frame_data_processed[sequence_name][f_data["frame_number"]] = f_data

            good_quality_sequences = set()
            for seq_data in sequence_data:
                if seq_data["viewpoint_quality_score"] > 0.5:
                    good_quality_sequences.add(seq_data["sequence_name"])

            for subset in ["train"]:
                for seq_name, frame_number, filepath in subset_lists_data[subset]:
                    if seq_name not in good_quality_sequences:
                        continue

                    if seq_name not in self.rotations:
                        self.rotations[seq_name] = []
                        self.category_map[seq_name] = c

                    mask_path = filepath.replace("images", "masks").replace(".jpg", ".png")

                    frame_data = frame_data_processed[seq_name][frame_number]

                    self.rotations[seq_name].append(
                        {
                            "filepath": filepath,
                            "R": frame_data["viewpoint"]["R"],
                            "T": frame_data["viewpoint"]["T"],
                            "focal_length": frame_data["viewpoint"]["focal_length"],
                            "principal_point": frame_data["viewpoint"]["principal_point"],
                            "mask": mask_path,
                            "txt": "a car",
                            "bbox": bbox_data[mask_path]
                        }
                    )

        for seq_name in self.rotations:
            seq_data = self.rotations[seq_name]
            cameras = PerspectiveCameras(
                        focal_length=[data["focal_length"] for data in seq_data],
                        principal_point=[data["principal_point"] for data in seq_data],
                        R=[data["R"] for data in seq_data],
                        T=[data["T"] for data in seq_data],
                    )

            normalized_cameras, _, _, _, _ = normalize_cameras(cameras)
            if aligncameras:
                normalized_cameras = centerandalign(cameras)

            if normalized_cameras == -1:
                print("Error in normalizing cameras: camera scale was 0")
                del self.rotations[seq_name]
                continue

            for i, data in enumerate(seq_data):
                self.rotations[seq_name][i]["R"] = normalized_cameras.R[i]
                self.rotations[seq_name][i]["T"] = normalized_cameras.T[i]
                self.rotations[seq_name][i]["R_original"] = torch.from_numpy(np.array(seq_data[i]["R"]))
                self.rotations[seq_name][i]["T_original"] = torch.from_numpy(np.array(seq_data[i]["T"]))

                # Make sure translations are not ridiculous
                if self.rotations[seq_name][i]["T"][0] + self.rotations[seq_name][i]["T"][1] + self.rotations[seq_name][i]["T"][2] > 1e5:
                    bad_seq = True
                    self.low_quality_translations.append(seq_name)
                    break

        for seq_name in self.low_quality_translations:
            if seq_name in self.rotations:
                del self.rotations[seq_name]

        self.sequence_list = list(self.rotations.keys())

        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0)
            ]
        )
        self.transformim = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0)
            ]
        )
        self.transformmask = transforms.Compose(
            [
                transforms.Resize(img_size // 8),
                transforms.ToTensor(),
            ]
        )

        self.num_images = num_images
        self.image_size = img_size
        self.normalize_cameras = normalize_cameras
        self.mask_images = mask_images
        self.drop_ratio = drop_ratio
        self.kernel_tensor = torch.ones((1, 1, 7, 7))
        self.repeat = repeat
        self.valid_ids = np.arange(0, len(self.rotations[self.sequence_list[self.single_id]]), skip).tolist()
        if split == 'test':
            self.valid_ids = list(set(np.arange(0, len(self.rotations[self.sequence_list[self.single_id]])).tolist()).difference(self.valid_ids))

        print(
            f"Low quality translation sequences, not used: {self.low_quality_translations}"
        )
        print(f"Data size: {len(self)}")

    def __len__(self):
        return (len(self.valid_ids))*self.repeat + (1 if self.addlen else 0)

    def _padded_bbox(self, bbox, w, h):
        if w < h:
            bbox = np.array([0, 0, w, h])
        else:
            bbox = np.array([0, 0, w, h])
        return square_bbox(bbox.astype(np.float32))

    def _crop_bbox(self, bbox, w, h):
        bbox = square_bbox(bbox.astype(np.float32))

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2
        extent = side_length / 2

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new(
                "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
            )
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
        return image_crop

    def __getitem__(self, index, specific_id=None, validation=False):
        sequence_name = self.sequence_list[self.single_id]

        metadata = self.rotations[sequence_name]

        if validation:
            drop_text = False
            drop_im = False
        else:
            drop_im = np.random.uniform(0, 1) < self.drop_ratio
            if not drop_im:
                drop_text = np.random.uniform(0, 1) < self.drop_txt
            else:
                drop_text = False

        size = self.image_size

        # sample reference ids
        listofindices = self.valid_ids.copy()
        max_diff = len(listofindices) // (self.num_images-1)
        if (index*self.skip) % len(metadata) in listofindices:
            listofindices.remove((index*self.skip) % len(metadata))
        references = np.random.choice(np.arange(0, len(listofindices)+1, max_diff), self.num_images-1, replace=False)
        rem = np.random.randint(0, max_diff)
        references = [listofindices[(x + rem) % len(listofindices)] for x in references]
        ids = [(index*self.skip) % len(metadata)] + references

        # special case to save features corresponding to ref image as part of model buffer
        if self.onlyref:
            ids = references + [(index*self.skip) % len(metadata)]
        if specific_id is not None:  # remove this later
            ids = specific_id

        # get data
        batch = self.get_data(index=self.single_id, ids=ids)

        # text prompt
        if self.modifier_token is not None:
            name = self.category[0] if self.categoryname is None else self.categoryname
            batch['txt'] = [f'photo of a {self.modifier_token} {name}' for _ in range(len(batch['txt']))]

        # replace with regularization image if drop_im
        if drop_im and self.addreg:
            select_id = np.random.randint(0, self.reglen)
            batch["image"] = [self.transformim(Image.open(f'{self.regimpath}/images/{select_id}.png').convert('RGB'))]
            batch['txt'] = [self.regcaptions[select_id]]
            batch["original_size_as_tuple"] = torch.ones_like(batch["original_size_as_tuple"])*1024

        # create camera class and adjust intrinsics for crop
        cameras = [PerspectiveCameras(R=batch['R'][i].unsqueeze(0),
                                      T=batch['T'][i].unsqueeze(0),
                                      focal_length=batch['focal_lengths'][i].unsqueeze(0),
                                      principal_point=batch['principal_points'][i].unsqueeze(0),
                                      image_size=self.image_size
                                      )
                   for i in range(len(ids))]
        for i, cam in enumerate(cameras):
            adjust_camera_to_bbox_crop_(cam, batch["original_size_as_tuple"][i, :2], batch["crop_coords"][i])
            adjust_camera_to_image_scale_(cam, batch["original_size_as_tuple"][i, 2:], torch.tensor([self.image_size, self.image_size]))

        # create mask and dilated mask for mask based losses
        batch["depth"] = batch["mask"].clone()
        batch["mask"] = torch.clamp(torch.nn.functional.conv2d(batch["mask"], self.kernel_tensor, padding='same'), 0, 1)
        if not self.mask_images:
            batch["mask"] = [None for i in range(len(ids))]

        # special case to save features corresponding to zero image
        if index == self.__len__()-1 and self.addlen:
            batch["image"][0] *= 0.

        return {"jpg": batch["image"][0],
                "txt": batch["txt"][0] if not drop_text else "",
                "jpg_ref": batch["image"][1:] if not drop_im else torch.stack([2*torch.rand_like(batch["image"][0])-1. for _ in range(len(ids)-1)], dim=0),
                "txt_ref": batch["txt"][1:] if not drop_im else ["" for _ in range(len(ids)-1)],
                "pose": cameras,
                "mask": batch["mask"][0] if not drop_im else torch.ones_like(batch["mask"][0]),
                "mask_ref": batch["masks_padding"][1:],
                "depth": batch["depth"][0] if len(batch["depth"]) > 0 else None,
                "filepaths": batch["filepaths"],
                "original_size_as_tuple": batch["original_size_as_tuple"][0][2:],
                "target_size_as_tuple": torch.ones_like(batch["original_size_as_tuple"][0][2:])*size,
                "crop_coords_top_left": torch.zeros_like(batch["crop_coords"][0][:2]),
                "original_size_as_tuple_ref": batch["original_size_as_tuple"][1:][:, 2:],
                "target_size_as_tuple_ref": torch.ones_like(batch["original_size_as_tuple"][1:][:, 2:])*size,
                "crop_coords_top_left_ref": torch.zeros_like(batch["crop_coords"][1:][:, :2]),
                "drop_im": torch.Tensor([1-drop_im*1.])
                }

    def get_data(self, index=None, sequence_name=None, ids=(0, 1)):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        category = self.category_map[sequence_name]
        annos = [metadata[i] for i in ids]
        images = []
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []
        txts = []
        masks = []
        filepaths = []
        images_transformed = []
        masks_transformed = []
        original_size_as_tuple = []
        crop_parameters = []
        masks_padding = []
        depths = []

        for counter, anno in enumerate(annos):
            filepath = anno["filepath"]
            filepaths.append(filepath)
            image = Image.open(osp.join(CO3D_DIR, filepath)).convert("RGB")

            mask_name = osp.basename(filepath.replace(".jpg", ".png"))

            mask_path = osp.join(
                    CO3D_DIR, category, sequence_name, "masks", mask_name
                )
            mask = Image.open(mask_path).convert("L")

            if mask.size != image.size:
                mask = mask.resize(image.size)

            mask_padded = Image.fromarray((np.ones_like(mask) > 0))
            mask = Image.fromarray((np.array(mask) > 125))
            masks.append(mask)

            # crop image around object
            w, h = image.width, image.height
            bbox = np.array(anno["bbox"])
            if len(bbox) == 0:
                bbox = np.array([0, 0, w, h])

            if self.bbox and counter > 0:
                bbox = self._crop_bbox(bbox, w, h)
            else:
                bbox = self._padded_bbox(None, w, h)
            image = self._crop_image(image, bbox)
            mask = self._crop_image(mask, bbox)
            mask_padded = self._crop_image(mask_padded, bbox)
            masks_padding.append(self.transformmask(mask_padded))
            images_transformed.append(self.transform(image))
            masks_transformed.append(self.transformmask(mask))

            crop_parameters.append(torch.tensor([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]).int())
            original_size_as_tuple.append(torch.tensor([w, h, bbox[2] - bbox[0], bbox[3] - bbox[1]]))
            images.append(image)
            rotations.append(anno["R"])
            translations.append(anno["T"])
            focal_lengths.append(torch.tensor(anno["focal_length"]))
            principal_points.append(torch.tensor(anno["principal_point"]))
            txts.append(anno["txt"])

        images = images_transformed
        batch = {
            "model_id": sequence_name,
            "category": category,
            "original_size_as_tuple": torch.stack(original_size_as_tuple),
            "crop_coords": torch.stack(crop_parameters),
            "n": len(metadata),
            "ind": torch.tensor(ids),
            "txt": txts,
            "filepaths": filepaths,
            "masks_padding": torch.stack(masks_padding) if len(masks_padding) > 0 else [],
            "depth": torch.stack(depths) if len(depths) > 0 else [],
        }

        batch["R"] = torch.stack(rotations)
        batch["T"] = torch.stack(translations)
        batch["focal_lengths"] = torch.stack(focal_lengths)
        batch["principal_points"] = torch.stack(principal_points)

        # Add images
        if self.transform is None:
            batch["image"] = images
        else:
            batch["image"] = torch.stack(images)
            batch["mask"] = torch.stack(masks_transformed)

        return batch

    @staticmethod
    def collate_fn(batch):
        """A function to collate the data across batches. This function must be passed to pytorch's DataLoader to collate batches.
        Args:
            batch(list): List of objects returned by this class' __getitem__ function. This is given by pytorch's dataloader that calls __getitem__
                         multiple times and expects a collated batch.
        Returns:
            dict: The collated dictionary representing the data in the batch.
        """
        result = {
            "jpg": [],
            "txt": [],
            "jpg_ref": [],
            "txt_ref": [],
            "pose": [],
            "original_size_as_tuple": [],
            "original_size_as_tuple_ref": [],
            "crop_coords_top_left": [],
            "crop_coords_top_left_ref": [],
            "target_size_as_tuple_ref": [],
            "target_size_as_tuple": [],
            "drop_im": [],
            "mask_ref": [],
        }
        if batch[0]["mask"] is not None:
            result["mask"] = []
        if batch[0]["depth"] is not None:
            result["depth"] = []

        for batch_obj in batch:
            for key in result.keys():
                result[key].append(batch_obj[key])
        for key in result.keys():
            if not (key == 'pose' or 'txt' in key or 'size_as_tuple_ref' in key or 'coords_top_left_ref' in key):
                result[key] = torch.stack(result[key], dim=0)
            elif 'txt_ref' in key:
                result[key] = [item for sublist in result[key] for item in sublist]
            elif 'size_as_tuple_ref' in key or 'coords_top_left_ref' in key:
                result[key] = torch.cat(result[key], dim=0)
            elif 'pose' in key:
                result[key] = [join_cameras_as_batch(cameras) for cameras in result[key]]

        return result


class CustomDataDictLoader(pl.LightningDataModule):
    def __init__(
        self,
        category,
        batch_size,
        mask_images=False,
        skip=1,
        img_size=1024,
        num_images=4,
        num_workers=0,
        shuffle=True,
        single_id=0,
        modifier_token=None,
        bbox=False,
        addreg=False,
        drop_ratio=0.5,
        jitter=False,
        drop_txt=0.1,
        categoryname=None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_dataset = Co3dDataset(category,
                                         img_size=img_size,
                                         mask_images=mask_images,
                                         skip=skip,
                                         num_images=num_images,
                                         single_id=single_id,
                                         modifier_token=modifier_token,
                                         bbox=bbox,
                                         addreg=addreg,
                                         drop_ratio=drop_ratio,
                                         drop_txt=drop_txt,
                                         categoryname=categoryname,
                                         )
        self.val_dataset = Co3dDataset(category,
                                       img_size=img_size,
                                       mask_images=mask_images,
                                       skip=skip,
                                       num_images=2,
                                       single_id=single_id,
                                       modifier_token=modifier_token,
                                       bbox=bbox,
                                       addreg=addreg,
                                       drop_ratio=0.,
                                       drop_txt=0.,
                                       categoryname=categoryname,
                                       repeat=1,
                                       addlen=True,
                                       onlyref=True,
                                       )
        self.test_dataset = Co3dDataset(category,
                                        img_size=img_size,
                                        mask_images=mask_images,
                                        split="test",
                                        skip=skip,
                                        num_images=2,
                                        single_id=single_id,
                                        modifier_token=modifier_token,
                                        bbox=False,
                                        addreg=addreg,
                                        drop_ratio=0.,
                                        drop_txt=0.,
                                        categoryname=categoryname,
                                        repeat=1,
                                        )
        self.collate_fn = Co3dDataset.collate_fn

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True
        )
