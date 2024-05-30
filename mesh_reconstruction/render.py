# modified from https://github.com/Profactor/continuous-remeshing
import nvdiffrast.torch as dr
import torch
from typing import Tuple

def _warmup(glctx, device=None):
    device = 'cuda' if device is None else device
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device=device, **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

glctx = dr.RasterizeGLContext(output_db=False, device="cuda")

class NormalsRenderer:
    
    _glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: Tuple[int,int],
            mvp = None,
            device=None,
            ):
        if mvp is None:
            self._mvp = proj @ mv #C,4,4
        else:
            self._mvp = mvp
        self._image_size = image_size
        self._glctx = glctx
        _warmup(self._glctx, device)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float   in [-1, 1]
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vert_col = (normals+1)/2 #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4
