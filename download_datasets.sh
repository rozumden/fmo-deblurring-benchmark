## TODO: insert your dataset path
MYPATH=/cluster/home/denysr/scratch/dataset/

# falling objects dataset
mkdir ${MYPATH}falling_objects
wget http://ptak.felk.cvut.cz/personal/rozumden/falling_imgs_gt.zip -P ${MYPATH}falling_objects
unzip ${MYPATH}falling_objects/falling_imgs_gt.zip -d ${MYPATH}falling_objects/
rm ${MYPATH}falling_objects/falling_imgs_gt.zip

# TbD-3D dataset
mkdir ${MYPATH}TbD-3D
wget http://ptak.felk.cvut.cz/personal/rozumden/TbD-3D-imgs.zip -P ${MYPATH}TbD-3D
unzip ${MYPATH}TbD-3D/TbD-3D-imgs.zip -d ${MYPATH}TbD-3D/
rm ${MYPATH}TbD-3D/TbD-3D-imgs.zip

# TbD dataset
mkdir ${MYPATH}TbD
wget http://ptak.felk.cvut.cz/personal/rozumden/TbD_benchmark.zip -P ${MYPATH}TbD
unzip ${MYPATH}TbD/TbD_benchmark.zip -d ${MYPATH}TbD/
rm ${MYPATH}TbD/TbD_benchmark.zip