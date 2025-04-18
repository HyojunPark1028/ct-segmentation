### CT Segmentation Project _(Notebook → Module)_

- Supports **pre‑split** folders: `.../train / val / test`.
- Uses **Dice + Focal Loss** combo during training.

```bash
!git clone <repo>
%cd ct-segmentation
!pip install -r requirements.txt
from src.train import train
train("configs/unet.yaml")
```

`metrics.csv` records train loss, val dice/IoU each epoch plus final test metrics.

---