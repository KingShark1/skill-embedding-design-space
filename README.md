# skill-embedding-design-space
Creating a approximation objective design space for skill embeddings for physically simulated characters.

DONE :
1. Download Isaac Gym from Nvidia's website : https://developer.nvidia.com/isaac-gym and follow the setup instructions. Make sure to check for correct installation by running the examples given inside IsaacGym installation.
2. Add env/tasks as is, for it is directly related to the experimentation, for now focus is on developing a working model. Adding more tasks would come later.

TODO :
1. Create the learning models for flow, diffusion, CPC. (Focusing on [FLOW-GAN](https://lilianweng.github.io/posts/2018-10-13-flow-models/#types-of-generative-models ) )
2.	Add the learned models in ds_se/run.py inside build_alg_runner().
3.  Convert model from [FLOW](https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py) to the form of your network builder.

The dissertation entitled “Design Space Of Skill Embeddings For Physically Simulated Characters” submitted by Akash Mishra, Arinjay Saraf, Manas Tiwari and Yuvraj Singh is a satisfactory account of the bonafide work done under my supervision is recommended towards the partial fulfilment for the award of Bachelor of Engineering in Computer Engineering degree by Devi Ahilya Vishwavidyalaya, Indore.

Click on the link to open the [Dissertation](https://docs.google.com/document/d/1blAdvtx8M_-O_sqLXbRyBfapGMWhyWlwH4B1sB3vU-s/edit?usp=sharing)
