# skill-embedding-design-space
Creating a approximation objective design space for skill embeddings for physically simulated characters.

DONE :
1. Download Isaac Gym from Nvidia's website : https://developer.nvidia.com/isaac-gym and follow the setup instructions. Make sure to check for correct installation by running the examples given inside IsaacGym installation.
2. Add env/tasks as is, for it is directly related to the experimentation, for now focus is on developing a working model. Adding more tasks would come later.

TODO :
1. Create the learning models for flow, diffusion, CPC. (Focusing on [FLOW-GAN](https://lilianweng.github.io/posts/2018-10-13-flow-models/#types-of-generative-models ) )
2.	Add the learned models in ds_se/run.py inside build_alg_runner().

