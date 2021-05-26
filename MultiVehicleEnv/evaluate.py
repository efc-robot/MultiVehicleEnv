import sys
import termios
import contextlib
from .environment import MultiVehicleEnv
import numpy as np
from typing import Callable, List, Optional, Sequence
import threading


T_Policy = Callable[[np.ndarray],int]
T_action = List[int]

class EvaluateWrap(object):
    def __init__(self, env:MultiVehicleEnv,
                       policy_list:Sequence[T_Policy]):
        self.env = env
        self.policy_list:Optional[Sequence[T_Policy]] = policy_list

        
    def run_test(self):
        n_obs = self.env.reset()
        for step_idx in range(1000000):
            if self.stop_signal:
                break
            n_action:T_action = []
            for obs,policy in zip(n_obs,self.policy_list):
                action = policy(obs)
                n_action.append(action)
            n_obs,reward,done,info = self.env.step(n_action)

    @contextlib.contextmanager
    def raw_mode(self, file):
        old_attrs = termios.tcgetattr(file.fileno())
        new_attrs = old_attrs[:]
        new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
        try:
            termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
            yield
        finally:
            termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)

    def main(self):
        decoder = {'e':0,'q':1,'z':2,'c':3}
        while True:
            cmd = input('waiting for cmd: ')
            self.stop_signal = False
            print(cmd)
            if cmd == 's':
                t= threading.Thread(target=self.run_test)
                t.setDaemon(True)
                t.start()
                print('start for keyboard ctrl')
                with self.raw_mode(sys.stdin):
                    try:
                        while True:
                            ch = sys.stdin.read(1)
                            # key esc is pressed
                            if ch == chr(27):
                                break
                            if ch in decoder.keys():
                                self.env.world.data_slot['key_direction'] = decoder[ch]
                            #print ('%s' % ch)
                        self.stop_signal = True
                    except (KeyboardInterrupt, EOFError):
                        pass
            if cmd == 'x':
                print("finished!")
                break
        