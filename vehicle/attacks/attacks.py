from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim

import yaml
from easydict import EasyDict

from ultralytics import YOLO


from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data import build_yolo_dataset, ClassificationDataset, build_dataloader
from ultralytics.utils import TQDM, emojis
from ultralytics.utils.torch_utils import select_device

# from ultralytics.models.classify. import ClassificationValidator

from nudt_ultralytics.callbacks.callbacks import callbacks_dict

class attacks:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        self.model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        for (event, func) in callbacks_dict.items():
            self.model.add_callback(event, func)
        
        self.model.overrides = cfg
        self.device = self.model.device
            
    def get_dataloader(self) -> torch.utils.data.DataLoader:
        if str(self.cfg.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
            data = check_det_dataset(self.cfg.data)
            dataset = build_yolo_dataset(self.cfg, data.get(self.cfg.split), self.cfg.batch, data, mode="val", stride=self.model.stride)
            dataloader = build_dataloader(dataset, self.cfg.batch, self.cfg.workers, shuffle=False, rank=-1, drop_last=self.cfg.compile, pin_memory=False)
        elif self.cfg.task == "classify":
            data = check_cls_dataset(self.cfg.data, split=self.cfg.split)
            dataset = ClassificationDataset(root=data.get(self.cfg.split), args=self.cfg, augment=self.cfg.augment, prefix=self.cfg.split)
            dataloader = build_dataloader(dataset, self.cfg.batch, self.cfg.workers, rank=-1)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.cfg.data}' for task={self.cfg.task} not found ❌"))

        return dataloader
    
    def get_desc(self) -> str:
        """Return a formatted string summarizing classification metrics."""
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input batch by moving data to device and converting to appropriate dtype."""
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.cfg.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def run_adv(self, args):
        dataloader = self.get_dataloader()
        desc = self.get_desc()
        bar = TQDM(dataloader, desc=desc, total=len(dataloader))
        for batch_i, batch in enumerate(bar):
            batch = self.preprocess(batch)
            if args.attack_method == 'pgd':
                adv_images = self.pgd(batch["img"], batch["cls"], eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations, random_start=args.random_start)
            elif args.attack_method == 'fgsm':
                adv_images = self.fgsm(batch["img"], batch["cls"], eps=args.epsilon)
            elif args.attack_method == 'cw':
                adv_images = self.cw(batch["img"], batch["cls"], c=1, kappa=0, steps=args.max_iterations, lr=0.01)
            elif args.attack_method == 'bim':
                adv_images = self.bim(batch["img"], batch["cls"], eps=args.epsilon, alpha=args.step_size, steps=args.max_iterations)
            elif args.attack_method == 'deepfool':
                adv_images, _ = self.deepfool(batch["img"], batch["cls"], steps=args.max_iterations, overshoot=0.02)
            else:
                raise ValueError('Invalid attach method!')
        
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(adv_images[0])
            pil_image.save(f'{self.cfg.save_dir}/adv_images_{batch_i}.jpg')
        
        
###################################################################################################################################################

    def pgd(self, images, labels, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        '''
        PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
        [https://arxiv.org/abs/1706.06083]

        Distance Measure : Linf
    
        Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''
        
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        
        if random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach().detach()
        
        for _ in range(steps):
            adv_images.requires_grad = True
            
            results = self.model(adv_images, stream=False)
            preds = [result.probs.data for result in results] # # Probs object for classification outputs
            preds = torch.stack(preds)
            
            outputs = preds.to(self.device)
            loss = loss_fn(outputs, labels)
            
            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
    
    def fgsm(self, images, labels, eps=8 / 255):
        '''
        FGSM in the paper 'Explaining and harnessing adversarial examples'
        [https://arxiv.org/abs/1412.6572]

        Distance Measure : Linf
    
        Arguments:
        eps (float): maximum perturbation. (Default: 8/255)

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''
        loss_fn = nn.CrossEntropyLoss()
        images.requires_grad = True
        
        results = self.model(images, stream=False)
        preds = [result.probs.data for result in results] # # Probs object for classification outputs
        preds = torch.stack(preds)
        
        outputs = preds.to(self.device)
        loss = loss_fn(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
    
    def cw(self, images, labels, c=1, kappa=0, steps=50, lr=0.01):
        '''
        CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
        [https://arxiv.org/abs/1608.04644]

        Distance Measure : L2
    
        Arguments:
        c (float): c in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

        .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''
        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=lr)

        for step in range(steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            results = self.model(adv_images, stream=False)
            # print(results)
            preds = [result.probs.data for result in results] # # Probs object for classification outputs
            preds = torch.stack(preds)
            
            outputs = preds.to(self.device)
            f_loss = self.f(outputs, labels, kappa).sum()

            cost = L2_loss + c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            # If the attack is not targeted we simply make these two values unequal
            condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))
    
    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    # f-function in the paper
    def f(self, outputs, labels, kappa):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs.to(self.device), dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs.to(self.device), dim=1)[0]

        return torch.clamp((real - other), min=-kappa)
        
        
        
    def bim(self, images, labels, eps=8 / 255, alpha=2 / 255, steps=10):
        '''
        BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
        [https://arxiv.org/abs/1607.02533]

        Distance Measure : Linf
        
        Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

        .. note:: If steps set to 0, steps will be automatically decided following the paper.

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''
        if steps == 0:
            steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        
        loss_fn = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        for _ in range(steps):
            images.requires_grad = True
            
            results = self.model(images, stream=False)
            preds = [result.probs.data for result in results] # # Probs object for classification outputs
            preds = torch.stack(preds)

            outputs = preds.to(self.device)
            loss = loss_fn(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

            adv_images = images + alpha * grad.sign()
            a = torch.clamp(ori_images - eps, min=0)
            b = (adv_images >= a).float() * adv_images + (adv_images < a).float() * a  # nopep8
            c = (b > ori_images + eps).float() * (ori_images + eps) + (b <= ori_images + eps).float() * b  # nopep8
            adv_images = torch.clamp(c, max=1).detach()

        return adv_images
    
    
    def deepfool(self, images, labels, steps=50, overshoot=0.02):
        '''
        'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
        [https://arxiv.org/abs/1511.04599]
        Distance Measure : L2
        Arguments:
            steps (int): number of steps. (Default: 50)
            overshoot (float): parameter for enhancing the noise. (Default: 0.02)
        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''
        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0
        
        adv_images = []
        for idx in range(batch_size):
            image = images[idx : idx + 1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < steps):
            for idx in range(batch_size):
                if not correct[idx]:
                    continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()
        return adv_images, target_labels

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        
        results = self.model(image, stream=False)
        preds = [result.probs.data for result in results] # # Probs object for classification outputs
        preds = torch.stack(preds)
        
        fs = preds.to(self.device)
        _, pre = torch.max(fs, dim=-1)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (
            torch.abs(f_prime[hat_L])
            * w_prime[hat_L]
            / (torch.norm(w_prime[hat_L], p=2) ** 2)
        )

        target_label = hat_L if hat_L < label else hat_L + 1

        adv_image = image + (1 + overshoot) * delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
        