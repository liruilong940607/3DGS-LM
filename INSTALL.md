
### Dependencies

```
pip install torch torchvision  # cuda12.8
pip install "numpy<2.0.0" plyfile tqdm ffmpeg imageio

cd submodules/diff-gaussian-rasterization/
pip install .

cd ../simple-knn/
pip install .
```

### Run
```
bash train.sh
```
