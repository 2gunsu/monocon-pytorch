## Changelog

### v1.0.3 (2022.09.10)
- Made ```torch.backend.cudnn.benchmark``` option configurable in ```monocon_config.py```. (**Issue #2**)
  ```python
  _C.USE_BENCHMARK = True
  ```
- Some code has been modified and added to enable video inference using kitti raw dataset.

### v1.0.2 (2022.08.30)
- A Transform class ```Resize3D``` for resize augmentation has been added. (**Issue #1**)

### v1.0.1 (2022.08.29)
- Visualization functions for 2D Boxes, 3D Boxes, and BEV have been added.

### v1.0.0 (2022.08.28)
- This repository is released.