from collections import deque
import random

class Replay_Buffer(object):
    def __init__(self, buffer_size=10e6, batch_size=1):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

    def __call__(self):
        return self.memory

    def store_transition(self, transition):
        self.memory.append(transition)

    def store_transitions(self, transitions):
        self.memory.extend(transitions)

    def get_batch(self, batch_size=None):
        b_s = batch_size or self.batch_size
        cur_men_size = len(self.memory)
        if cur_men_size < b_s:
            return random.sample(list(self.memory), cur_men_size)
        else:
            return random.sample(list(self.memory), b_s)

    def memory_state(self):
        return {"buffer_size": self.buffer_size,
                "current_size": len(self.memory),
                "full": len(self.memory)==self.buffer_size}

    def empty_transition(self):
        self.memory.clear()

if __name__ == '__main__':
    import numpy as np
    replay_buffer = Replay_Buffer(buffer_size=4)
    print(replay_buffer.memory_state())
    replay_buffer.store_transition([1, 2, 3, 4, False])
    print(replay_buffer.memory_state())
    replay_buffer.store_transition([2, 2, 3, 4, False])
    print(replay_buffer.memory_state())
    replay_buffer.store_transition([3, 2, 3, 4, True])
    print(replay_buffer.memory_state())
    print(replay_buffer())

    replay_buffer.store_transition([4, 2, 3, 4, True])
    print(replay_buffer.memory_state())
    print(replay_buffer())

    replay_buffer.store_transitions([[5, 2, 3, 4, False],
                                     [6, 2, 3, 4, True]])
    print(replay_buffer.memory_state())
    print(replay_buffer())

    batch = replay_buffer.get_batch(3)
    print("batch", batch)
    transpose_batch = list(zip(*batch))
    print("transpose_batch", transpose_batch)
    s_batch = np.array(transpose_batch[0])
    a_batch = list(transpose_batch[1])
    r_batch = list(transpose_batch[2])
    next_s_batch = list(transpose_batch[3])
    done_batch = np.array(transpose_batch[4])
    print("s_batch", s_batch)
    print("a_batch", a_batch)
    print("r_batch", r_batch)
    print("next_s_batch", next_s_batch)
    print("done_batch", done_batch)
    print((1-done_batch)*s_batch)

    replay_buffer.empty_transition()
    print(replay_buffer.memory_state())
    print(replay_buffer())
