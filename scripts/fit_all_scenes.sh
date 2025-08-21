root_360v2="/path/to/360_v2"
root_db="/path/to/db"
root_tandt="/path/to/tandt"

# 360v2 outdoor
SCENE_LIST="bicycle garden stump flowers treehill"
for scene in $SCENE_LIST;
do
    echo "run scene $scene for 3DGS+Ours"
    python train.py \
    -s $root_360v2/$scene \
    --root_out outputs/$scene \
    --exp_name "3dgs_lm_$scene" \
    --eval \
    --images "images_4" \
    --resolution 1 \
    --image_subsample_size 25 \
    --image_subsample_n_iters 4 \
    --image_subsample_frame_selection_mode "strided" \
    --num_sgd_iterations_before_gn 20000 \
    --perc_images_in_line_search 0.3 \
    --pcg_rtol 5e-2 \
    --pcg_max_iter 8 \
    --min_trust_region_radius 1e-4 \
    --trust_region_radius 1e-3 \
    --max_trust_region_radius 1e-2 \
    --iterations 5 \
    --test_iterations 5 \
    --save_iterations 5
done

# 360v2 indoor
SCENE_LIST="kitchen bonsai room counter"
for scene in $SCENE_LIST;
do
    echo "run scene $scene for 3DGS+Ours"
    python train.py \
    -s $root_360v2/$scene \
    --root_out outputs/$scene \
    --exp_name "3dgs_lm_$scene" \
    --eval \
    --images "images_2" \
    --resolution 1 \
    --image_subsample_size 25 \
    --image_subsample_n_iters 4 \
    --image_subsample_frame_selection_mode "strided" \
    --num_sgd_iterations_before_gn 20000 \
    --perc_images_in_line_search 0.3 \
    --pcg_rtol 5e-2 \
    --pcg_max_iter 8 \
    --min_trust_region_radius 1e-4 \
    --trust_region_radius 1e-3 \
    --max_trust_region_radius 1e-2 \
    --iterations 5 \
    --test_iterations 5 \
    --save_iterations 5
done

# db
SCENE_LIST="playroom drjohnson"
for scene in $SCENE_LIST;
do
    echo "run scene $scene for 3DGS+Ours"
    python train.py \
    -s $root_db/$scene \
    --root_out outputs/$scene \
    --exp_name "3dgs_lm_$scene" \
    --eval \
    --image_subsample_size 25 \
    --image_subsample_n_iters 4 \
    --image_subsample_frame_selection_mode "random" \
    --num_sgd_iterations_before_gn 20000 \
    --perc_images_in_line_search 0.3 \
    --pcg_rtol 5e-2 \
    --pcg_max_iter 8 \
    --min_trust_region_radius 1e-4 \
    --trust_region_radius 1e0 \
    --max_trust_region_radius 1e4 \
    --iterations 5 \
    --test_iterations 5 \
    --save_iterations 5
done

# tandt
SCENE_LIST="truck train"
for scene in $SCENE_LIST;
do
    echo "run scene $scene for 3DGS+Ours"
    python train.py \
    -s $root_tandt/$scene \
    --root_out outputs/$scene \
    --exp_name "3dgs_lm_$scene" \
    --eval \
    --image_subsample_size 60 \
    --image_subsample_n_iters 3 \
    --image_subsample_frame_selection_mode "random" \
    --num_sgd_iterations_before_gn 20000 \
    --perc_images_in_line_search 0.3 \
    --pcg_rtol 5e-2 \
    --pcg_max_iter 8 \
    --min_trust_region_radius 1e-4 \
    --trust_region_radius 1e0 \
    --max_trust_region_radius 1e4 \
    --iterations 5 \
    --test_iterations 5 \
    --save_iterations 5
done