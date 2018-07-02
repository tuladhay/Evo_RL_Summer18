import copy

class PD:
    def __init__(self, target, delta):
        self.kp = 0.0
        self.kd = 0.0
        self.target = target
        self.delta = delta
        self.previous_error = 0.0
        self.fitness = 0.0

    def compute_pd_output(self, position, noise=None):
        '''
        :param position: measured position and velocity of the block
        :param noise: signal/sensor noise
        :return: PD control output
        '''
        error = position - self.target
        # print("error = " + str(error))
        delta_error = error - self.previous_error
        output = -self.kp*error - self.kd*(delta_error/self.delta)
        self.previous_error = error
        # print("output = " + str(output))
        return output

    def set_gains(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def set_target(self, target):
        self.target = target

    def set_delta(self, delta):
        self.delta = delta
