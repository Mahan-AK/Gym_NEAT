#!/anaconda3/bin/python
import multiprocessing
import numpy as np
import os
import pickle
import neat
import gym

env_name = "MountainCar-v0"

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    runs_per_net = 3
    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make(env_name)

        observation = env.reset()

        done = False
        max_pos = 0.0

        while not done:
            action = np.argmax(net.activate(observation))
            observation, _, done, _ = env.step(action)

            max_pos = observation[0] if observation[0] > max_pos else max_pos

        fitnesses.append(max_pos)

    return np.min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    config_path = "config_MountainCar"

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_path)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    with open('winner_MountainCar', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

if __name__ == '__main__':
    run()
