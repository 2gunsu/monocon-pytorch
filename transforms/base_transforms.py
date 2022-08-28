from typing import Dict, List, Any


class BaseTransform:
    def __init__(self,
                 change_img: bool,
                 change_metas: bool,
                 change_calib: bool,
                 change_label: bool):
        
        self._change_img = change_img
        self._change_metas = change_metas
        self._change_calib = change_calib
        self._change_label = change_label
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    
    def __repr__(self):
        args_dict = {k: v for k, v in self.__dict__.items() 
                     if not callable(getattr(self, k)) and not k.startswith('_')}
        
        cls_name = self.__class__.__name__
        args_str = '('
        
        if len(args_dict) > 0:
            for idx, (k, v) in enumerate(args_dict.items()):
                if idx != (len(args_dict) - 1):
                    args_str += f'{k}={v}, '
                else:
                    args_str += f'{k}={v})'
        else:
            args_str += ')'
        return (cls_name + args_str)
    

class Compose:
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms
    
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict
    