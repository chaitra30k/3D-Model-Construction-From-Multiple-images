import torch
from models.superglue import SuperGlue

def match_superglue(desc0, desc1, kpts0, kpts1, scores0, scores1, device='cuda'):
    model = SuperGlue({}).to(device)
    model.eval()
    data = {
        'keypoints0': torch.from_numpy(kpts0).unsqueeze(0).to(device),
        'keypoints1': torch.from_numpy(kpts1).unsqueeze(0).to(device),
        'descriptors0': torch.from_numpy(desc0).unsqueeze(0).to(device),
        'descriptors1': torch.from_numpy(desc1).unsqueeze(0).to(device),
        'scores0': torch.from_numpy(scores0).unsqueeze(0).to(device),
        'scores1': torch.from_numpy(scores1).unsqueeze(0).to(device),
    }
    with torch.no_grad():
        result = model(data)
    matches = result['matches0'][0].cpu().numpy()
    return matches 