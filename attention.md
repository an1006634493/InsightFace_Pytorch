1. When install `bcolz`, just directly use `pip install bcolz`, don not use `setup.py`.
2. When install `mxnet_cu90`, do not use `pip install mxnet_cu90==1.2.1`，otherwise when you `import mxnet`, there will be `Illegal instruction (core dumped)`. Please use `pip install mxnet-cu90==1.3.1b20181004`.
3. After download `model_ir_se50.pth`, put it in `\work_space\save` and rename it as `model_final.pth`. Before this step, just detele all files in `\work_space` and create new folders with the same names.
4. Every time you run `face_verify.py` or `infer_on_video.py`, add empty arg `--update` to make sure it deals with the right faces in facebank.
5. Maybe the video generating process has some bug. Just use `frame` to visualize performance. Don't use `cv2.imshow` as it may lead to crash, use `cv2.imwrite` instead and open it with other softwares.
