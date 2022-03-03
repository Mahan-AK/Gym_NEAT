#!/anaconda3/bin/python
import multiprocessing
import numpy as np
import os
import pickle
import neat
import gym

env_name = "BipedalWalker-v3"

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    runs_per_net = 2
    fitness = 0.0
    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make(env_name)

        observation = env.reset()
        done = False

        while not done:
            action = net.activate(observation)
            observation, reward, done, _ = env.step(action)

            fitness += reward

        fitnesses.append(fitness)
        
    return np.min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    config_path = "config_BipedalWalker"

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

    with open('winner_BipedalWalker', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

if __name__ == '__main__':
    run()
