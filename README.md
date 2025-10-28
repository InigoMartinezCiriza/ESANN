# Bioinspired RNN for Time-Dependant RL Tasks

This repository contains the code associated to the work **"Bioinspired RNN for Time-Dependant RL Tasks"**, for the ESANN 2026 conference.

## Description
Understanding the brain mechanisms underlying cognition and learning is a major challenge in neuroscience. This work explores the use of Recurrent Neural Networks (RNN), based on modified GRU units and trained with Reinforcement Learning (RL), as models for specific brain regions involved in cognitive tasks. The aim is to incorporate bioinspired features into the network architecture and neural units to reflect principles observed in decision-making structures. When used in a RL environment requiring temporal memory, the model demonstrates the ability to solve complex tasks, highlighting its potential to bridge computational abstraction and biological neural processes.

It is trained in a Economic Choice Environmnt based on a real-life experiment (Padoa-Schioppa C, Assad JA. Neurons in the orbitofrontal cortex encode economic value. Nature. 2006 May 11;441(7090):223-6).

The project applies the REINFORCE algorithm with baseline to train an Actor-Critic network using modified GRU cells to mimic biological processes. Requires Python 3.12.3

## Structure
- Figures: contains the figures used in the article.
- Saved: contains the data obtained from training. Some files are compressed due to its large size.
    - Checkpoints: contains the masks and the network state for each training stage.
    - Outputs: contains the data (rewards, losses, choices...) of each training stage.
- Modules: contains the modules implemented: training algorithm, network structure and environments description.
- Notebooks: contains the training of the agents and the obtention of the results.

## License
Academic work. For use and redistribution, please consult the author.

## Author
Iñigo Martínez Ciriza and Carlos María Alaíz Gudín.
