# README

### Pix2Pix

train example is shown in jupyter notebook file
data and code working folder example:

  -Pix2Pix_for_edges2shoes.ipynb
  
  -edges2shoes
  
  --train
  
  --val
  
It will build save_models and images_wg folders and sample images and models automatically.
If you want to use models, please order the files as:
  
  -Pix2Pix_for_edges2shoes.ipynb
  
  -edges2shoes
  
  --train
  
  --val
  
  -save_models
  
  --Generate_1
  
  ---discriminator_80.pth
  
  ---generator_80.pth

and in .ipynb, change epoch = 80, note that the suffix means the training epochs

### CycleGAN lsl part

Train example is shown in jupyter notebook file
The CycleGAN-Unet file uses Unet as generator and the other uses Residualblocks as generator 
folder example:

  -CycleGAN-Unet.ipynb

  -CycleGAN.ipynb

  -Generate1

  --teA

  --teB
  
  --trainA
  
  --trainB

note that we cut and process the edges2shoes data into Generate1 folder
It will build save_models/save_models_U and images/images_U folders and sample images and models automatically.
If you want to use models, please order the files as:

  -CycleGAN-Unet.ipynb

  -CycleGAN.ipynb

  -Generate1

  --teA

  --teB
  
  --trainA
  
  --trainB
  
  -save_models
  
  --Generate1
  
  ---D_A_99.pth
  
  ---G_AB_99.pth
  
  ---D_B_99.pth
  
  ---G_BA_99.pth
  
  -saved_models_U
  
  --Generate1
  
  ---D_A_49.pth
  
  ---G_AB_49.pth
  
  ---D_B_49.pth
  
  ---G_BA_49.pth
 and change the epoch in .ipynb to Corresponding epoch
  
### CycleGAN

main.py:  This is the main file for training or testing the model.

network.py:  This file defines some useful layers used in the G and D models

and the initialization function.

discriminators.py:  This file defines the Discriminator network.

generators.py:  This file defines the Generator network with 9 resnet blocks.

model.py:  This file defines the cycleGAN model and training steps.

utils.py:  This file defines some useful functions.

test.py:  This file is used to test the model output.

results:  The horse2zebra results generated by CycleGAN.(Real - Generated - Reconstructed)

![result](C:\Users\徐锦成\Desktop\result.jpg)

- To run training

```
python main.py --training True
```

- To run testing

```
python main.py --test True --batch_size 1
```

We don't include the training data and the trained model in the submitted file since it takes too much space, we will upload it to the PKU disk and will share the link in the email. If the TA wants to test the result, the data and trained model should be named correctly and be put under the right folder. If there is any question about testing the result, please contact us.
