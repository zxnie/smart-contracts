import numpy as np


# Channel class
class Channel:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.state_success = 0
        self.state_error = 1
        self.channel_transitions = {
            0: {0: 0.9, 1: 0.1},
            1: {0: 0.3, 1: 0.7},
        }
        self.state_current = 0

    def transit_channel_state(self):
        self.state_current = self.rng.choice([self.state_success, self.state_error],
                                             p=[self.channel_transitions[self.state_current][self.state_success],
                                                self.channel_transitions[self.state_current][self.state_error]])
        if self.state_current == 0:
            return 'success'
        else:
            return 'fail'


# Node class
class WirelessNode:
    def __init__(self, id, seed):
        self.id = id
        self.channel = Channel(seed)
        self.sent_messages = 0
        self.latencies = []

    def send_message(self, env, transmission_time):
        env.process(self.transmit(env, transmission_time))

    def transmit(self, env, transmission_time):
        start_time = env.now
        while True:
            yield env.timeout(transmission_time)
            state = self.channel.transit_channel_state()
            if state == 'success':
                self.sent_messages += 1
                break
        end_time = env.now
        self.latencies.append(end_time - start_time)

    def reset_stats(self):
        self.sent_messages = 0
        self.latencies = []
