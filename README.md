### A Curious Model Is All You Need

Reinforcement learning heavily relies on environments to provide correct rewards for the task at hand. However, these extrinsic rewards need to be manually handengineered, individually for each environment and can be highly sparse in nature. Curiosity is a type of dense intrinsic reward function which uses future prediction error as reward signal. We propose a generative model-based recurrent neural network, trained using self-supervision, to provide this reward signal. We show that, purely curiosity-driven learning, i.e. without any extrinsic rewards, is sufficient to solve certain ALE environment.

