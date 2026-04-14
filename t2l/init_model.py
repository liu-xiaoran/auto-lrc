from . import t2l
from .mtl.wrapper import load_mtl_model

model_name, model_idx, demucs_model = 'mdx_extra', 1, None
mtl_model = 'MTL'

if __name__ == '__main__':
    demucs_model = t2l.load_demucs_model(model_name, model_idx)
    print('Model inited,', model_name, model_idx)
    mtl_model = load_mtl_model(method=mtl_model, cuda=True)
