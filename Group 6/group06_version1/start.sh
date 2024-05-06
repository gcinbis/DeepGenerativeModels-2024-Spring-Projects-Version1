pip install -r requirements.txt
git clone https://github.com/mit-han-lab/data-efficient-gans/
cp train.py data-efficient-gans/DiffAugment-stylegan2-pytorch/
cp kd_loss.py data-efficient-gans/DiffAugment-stylegan2-pytorch/training/
cp CLIP.py data-efficient-gans/DiffAugment-stylegan2-pytorch/training/
cp download_dataset.py data-efficient-gans/DiffAugment-stylegan2-pytorch/
cd data-efficient-gans/DiffAugment-stylegan2-pytorch
python download_dataset.py --path 100-shot-obama
python dataset_tool.py --source 100-shot-obama --dest 100-shot-obama-dataset