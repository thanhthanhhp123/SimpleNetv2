datapath=/content/Dataset
datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python /content/SimpleNetv2/main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_mvtec \
--log_project MVTecAD_Results \
--results_path results \
--run_name run \
net \
-le layer3 \
-le layer4 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 5 \
--embedding_size 256 \
--gan_epochs 2 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 8 \
--resize 329 \
--imagesize 288 "${dataset_flags[@]}" mvtec $datapath
