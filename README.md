# OpenAI Gym + NEAT Algorithm

In this project I aimed to solve different [OpenAI Gym](https://gym.openai.com/) environments using [NEAT (Neuroevolution of augmenting topologies)](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) Algorithm.

## Requirements

The scripts are all written in python3. You will also need to install ```gym``` to simulate the environments and ```neat-python``` for the NEAT implementation:

```shell
$ pip install gym neat-python
```

## Usage

All the files required to train and run the models for each environment can be found in the corresponding folder. You can change NEAT parameters such as population size, fitness threshhold, default genome architecture,.. in the .config file.

```shell
$ cd Acrobot
$ nano config_Acrobot # Play around with the parameters!
$ python Acrobat.py
```

This will run the NEAT algorithm and save the best individual as winner_ENVIRONMENT. You can now visualize how this individual performs on different instances of the environment:

```shell
$ python test_Acrobot.py
```