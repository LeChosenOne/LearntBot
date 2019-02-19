import sys
import math
import os
import queue
import threading
import random
import pickle
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

class LearntBot(BaseAgent):

    # fitness function for genomes
    def eval_genomes(self, genomes, config):
        self.max_individuals = len(genomes)
        for genome_id, genome in genomes:
            # store this genome's ANN for use by the bot, eventually come up with a fitness
            genome_NN = neat.nn.FeedForwardNetwork.create(genome, config)
            if self.generation % 50 == 0: # every 5 gens, pickle a neural net
                pickle.dump(genome_NN,  open( "neural_net", "wb" ))
            self.net_queue.put(genome_NN)
            # wait for best fitness from this genome
            best_fitness = self.fitness_queue.get()
            genome.fitness = best_fitness

    # drives NEAT algorithm
    def run_neat(self):
        import neat
        global neat
        # Load configuration.
        local_dir = os.path.dirname(__file__)
        config_file = os.path.join(local_dir, 'neat.config')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)
        #p.add_reporter(neat.StdOutReporter(True))
        #stats = neat.StatisticsReporter()
        #p.add_reporter(stats)
        #p.add_reporter(neat.Checkpointer(5))
        best_genome = p.run(self.eval_genomes, 100000)
        best_genome_neural_network = neat.nn.FeedForwardNetwork.create(best_genome, config)
        pickle.dump(best_genome_neural_network,  open( "best_neural_net", "wb" ))

    # set up learnt bot
    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.frame = 0
        self.generation = -1
        self.max_individuals = 0
        self.individual = -1

        # Queues for communicating models/data between NEAT and bot input/output
        self.initial_distance = 0
        self.percent_progress = 0
        self.net_queue = queue.Queue()
        self.fitness_queue = queue.Queue()

        # fire off thread for NEAT algo running concurrently with bot input/output
        neat_thread = threading.Thread(target = self.run_neat, args=())
        neat_thread.daemon = True
        neat_thread.start()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # 100 frames per individual, 10 individuals per generation, 100 generations, or reset before a goal happens
        if self.frame == (100 + self.generation*2) or self.frame == 0 or (-893 <= packet.game_ball.physics.location.x <= 893 and packet.game_ball.physics.location.y > 5115):
            if self.frame != 0:
                self.fitness_queue.put(self.percent_progress)
            self.frame = 0
            self.individual += 1
            self.percent_progress = 0.0
            # set a random game state so the bot doesn't start memorizing saves
            car_state = CarState(jumped=False, double_jumped=False, boost_amount=100,
                     physics=Physics(velocity=Vector3(x=0,y=0, z = 0), rotation=Rotator(0, math.pi * -1.5 + random.uniform(-0.5,0.5), 0),
                     angular_velocity=Vector3(0, 0, 0), location=Vector3(x=random.uniform(-1000,1000), y=random.uniform(2000,4000), z=0)))
            ball_state = BallState(Physics(location=Vector3(x=random.uniform(-800,800), y=random.uniform(-1000,1000), z=0), rotation=Rotator(0, 0, 0),
                     velocity=Vector3(x=0,y=random.uniform(1000,1500),z=random.uniform(0,500))))
            game_state = GameState(ball=ball_state, cars={self.index: car_state},)
            self.set_game_state(game_state)
            if self.individual == self.max_individuals:
                self.individual = 0
                self.generation += 1
            # wait for NEAT to have the next network ready
            self.network = self.net_queue.get()

        # run all code below on every frame
        self.frame += 1
        ball = packet.game_ball
        bot = packet.game_cars[self.index]

        # if first frame, record the original ball distance, otherwise try to get best distance to ball
        distance = math.sqrt((bot.physics.location.x - ball.physics.location.x)**2 + (bot.physics.location.y - ball.physics.location.y)**2 + (bot.physics.location.z - ball.physics.location.z)**2)
        if self.frame == 10:
            self.initial_distance = distance
        if self.frame >= 10:
            self.percent_progress = max(self.percent_progress, (self.initial_distance - distance)/self.initial_distance)

        # feature input vector that is used to train our neural network
        input = []
        input.append(bot.physics.location.x)
        input.append(bot.physics.location.y)
        input.append(bot.physics.location.z)
        #input.append(bot.physics.rotation.pitch)
        input.append(bot.physics.rotation.yaw)
        #input.append(bot.physics.rotation.roll)
        input.append(bot.physics.velocity.x)
        input.append(bot.physics.velocity.y)
        input.append(bot.physics.velocity.z)
        input.append(bot.physics.angular_velocity.x)
        input.append(bot.physics.angular_velocity.y)
        #input.append(bot.physics.angular_velocity.z)
        input.append(bot.boost)
        input.append(bot.jumped)
        input.append(bot.double_jumped)
        input.append(bot.has_wheel_contact)
        input.append(ball.physics.location.x)
        input.append(ball.physics.location.y)
        input.append(ball.physics.location.z)
        #input.append(ball.physics.rotation.pitch)
        #input.append(ball.physics.rotation.yaw)
        #input.append(ball.physics.rotation.roll)
        input.append(ball.physics.velocity.x)
        input.append(ball.physics.velocity.y)
        input.append(ball.physics.velocity.z)
        #input.append(ball.physics.angular_velocity.x)
        #input.append(ball.physics.angular_velocity.y)
        #input.append(ball.physics.angular_velocity.z)

        # use self.network (from NEAT) with input to derive a usable output
        def sigmoid(x):
            if x >= 0:
                z = math.e ** -x
                return 1. / (1. + z)
            else:
                z = math.e ** x
                return z / (1. + z)

        def activate(x):
            return sigmoid(x/3000) * 2 - 1
        output = self.network.activate(input)
        self.controller_state.throttle = activate(output[0])
        self.controller_state.steer = activate(output[1])
        self.controller_state.boost = True if activate(output[2]) > 0.75 else False
        #self.controller_state.pitch = activate(output[2])
        #self.controller_state.yaw = activate(output[3])
        #self.controller_state.roll = activate(output[4])
        #self.controller_state.jump = True if activate(output[5]) > 0.75 else False
        #self.controller_state.handbrake = True if activate(output[7]) > 0.75 else False
        if self.frame >= 10:
            self.renderer.begin_rendering()
            self.renderer.draw_string_2d(20,20, 2, 2, "Generation: " + str(self.generation) + " [Bot: " + str(self.individual) + "/" + str(self.max_individuals-1) + " | " + str(self.frame) + " frames]", self.renderer.white())
            self.renderer.draw_string_2d(20,50, 2, 2, "Best percent progress: " + str(round(self.percent_progress *100)) + "%", self.renderer.white())
            self.renderer.draw_string_2d(20,80, 2, 2, "Current percent progress: " + str(round((self.initial_distance - distance)/self.initial_distance*100)) + "%", self.renderer.white())
            self.renderer.draw_string_2d(20,110 , 2, 2, "Steer: " + str(self.controller_state.steer), self.renderer.white())
            self.renderer.draw_string_2d(20,140, 2, 2, "Throttle: " + str(self.controller_state.throttle), self.renderer.white())
            self.renderer.draw_string_2d(20,170, 2, 2, "Boost: " + str(self.controller_state.boost), self.renderer.white())
            #self.renderer.draw_string_2d(20,170, 2, 2, "Pitch: " + str(self.controller_state.pitch), self.renderer.white())
            #self.renderer.draw_string_2d(20,200, 2, 2, "Yaw: " + str(self.controller_state.yaw), self.renderer.white())
            #self.renderer.draw_string_2d(20,230, 2, 2, "Roll: " + str(self.controller_state.roll), self.renderer.white())
            #self.renderer.draw_string_2d(20,260, 2, 2, "Powerslide: " + str(self.controller_state.handbrake), self.renderer.white())
            #self.renderer.draw_string_2d(20,290, 2, 2, "Jump: " + str(self.controller_state.jump), self.renderer.white())
            self.renderer.end_rendering()
        #print(self.controller_state.__dict__)
        return self.controller_state
