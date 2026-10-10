# DiffEdit paper implemenmtation
This notebook is an implementation of [this paper](http://https://arxiv.org/abs/2210.11427). It was a "homework" as part of Fast.ai's practical deep learning course part 2.
When i got stuck I ocationally referenced [this](https://gist.github.com/kevinbird15/00b7ade94d25e8a1dde3c151b8631ea4) implementation, but have also built upon it. For example, during decoding we now use the noisy latents from the ecoding process, as opposed to copying the latents of the original image.

## Short description of DiffEdit
This method aims to modify specific sections of an image based on a given prompt by first automatically generating a mask, then running diffusion, at each timestep, replacing the areas outside the mask with the original image with noise from that timestep.
Thus, there are 3 major steps:
1. Generate a mask by:
   - adding some noise to the original image
   - estimating the noise:
        - once conditioned on the text prompt we want to achieve (querry)
        - once on a description of the oiginal image (reference, can also be empty)
   - taking the difference of the estimated noises and binarizing.
2. Encoding:
    - add noise to the original image over some timesteps
    - we use **r * num_inference_steps** steps  for encoding, where r (in [0, 1]) is defined as the ecoding ratio
3. Decoding:
    - take the encoded image from the previous step and predict the noise conditional on the querry
    - at each timestep, t, only keep the predictions that fit within the mask from step 1
    - replace the rest with the noised initial image from the ecoding process at the coresponding timestep t.
