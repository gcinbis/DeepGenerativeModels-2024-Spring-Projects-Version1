echo "Creating the training dataset directory"
mkdir data data/hr data/lr

echo "Downloading the training dataset"

echo "Downloading DIV2K dataset..."
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

echo "Unzipping DIV2K dataset..."
unzip DIV2K_train_HR.zip 
rm DIV2K_train_HR.zip
mv DIV2K_train_HR/* data/hr

echo "Downloading Flickr2K dataset..."
wget https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar

echo "Unzipping Flickr2K dataset..."
tar -xvf Flickr2K.tar
rm Flickr2K.tar
mv Flickr2K/* data/hr

echo "Generating low res images"
python resize_4.py --folder data/hr --save_path data/lr

echo "Preprocessing the dataset"
python prepare_dataset.py

echo "Downloading the evaluation data that consists of Manga109 (only manga is included for version 1)"
mkdir data/evaluation data/evaluation/hr data/evaluation/lr data/evaluation/hr/manga109 data/evaluation/lr/manga109 

echo "Downloading Manga109 dataset..."
gdown --id 1jw7QNbzea9SUq4IoRCO2q44TIyEXOl-f
unzip MANGA109.zip
rm MANGA109.zip
mv MANGA109/* data/evaluation/hr/manga109
python resize_4.py --folder data/evaluation/hr/manga109 --save_path data/evaluation/lr/manga109

echo "Downloading VGG network"
gdown https://drive.google.com/file/d/1henrktM4Cw9hJIJBDEObAzl-eCbpzNaJ/view?usp=drive_link

echo "Downloading the pre-trained model"
gdown --id 1WHPkBKkA2Bm19PnbOi19F9yp9YoN2FCN
echo "All done!"