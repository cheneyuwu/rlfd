# Undergrad Research Project
Work Record<br>
December 15, 2018

## *TODO* From Last Week
- Build the peg in hole environment by extending the robosuite environment
- We did not talk much about the code structure so I just used OpenAI's code base.
- Explore the structure of the medthods. and where does those methods fit into

## *Summary of Last Week*
- Please see below.

## Implementations and Experiments
- Mainly working on understanding the source code of DDPG(+HER+DEMO) from OpenAI, and constructing my own code base to run experiments on robosuite tasks (PegInHole). I have trained and tested my implementation of HER on some environments provided in OpenAI gym (since their environments already provides the goal state and is easy to test). 
- Currently I am still refining the code to separate each module and add implementation for the ensemble of q functions (our first proposed improvements). Code has been uploaded to GitHub and I can share it with you later. I mostly followed the structure of OpenAI's implementation of DDPG and re-used most of their utility functions.
## Papers
- For model based methods, I watched the lecture you recommended in your last email, also went through some papers listed in OpenAI's spinning up. Since we did not talk about this last time, I hope I can discuss the ideas with you before I start implementing any of them, or using them to improve our current ideas.
## Simulators and Tasks
- I quickly went through the documentations and played with the Nvidia PhysX you mentioned in the email. However, currently most of my tests were run on existing environments from gym or robosuite (mojuco based), let's discuss whether we should extend these existing environments or build our own. One problem with this might be the demonstration data. 
- In addition, I also want to confirm with you about what goal state and reward function I should use for the PegInHole environment.

## *TODO*
- Build the peg in hole environment by extending the robosuite environment
- We did not talk much about the code structure so I just used OpenAI's code base.
- Explore the structure of the medthods. and where does those methods fit into