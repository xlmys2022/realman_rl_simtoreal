"""Read-only motion statistics shared by training and evaluation."""
import numpy as np


class MotionStats:
    """Read-only SI motion/action diagnostics; no guidance or control output."""
    def __init__(self):
        self.steps=self.actual_valid=self.command_valid=self.actual_toward=self.command_toward=0
        self.improving=self.reversed=self.pairs=0
        self.actual_path=self.command_path=0.
        self.previous=None

    def observe(self,before,after,goal,command_before,command_after,action,done):
        action=np.asarray(action,dtype=float)
        for error,kind in [(np.subtract(goal,before),'actual'),(np.subtract(goal,command_before),'command')]:
            if np.linalg.norm(action)>1e-8 and np.linalg.norm(error)>1e-12:
                setattr(self,kind+'_valid',getattr(self,kind+'_valid')+1)
                setattr(self,kind+'_toward',getattr(self,kind+'_toward')+int(action@error>0))
        self.steps+=1
        self.actual_path+=float(np.linalg.norm(np.subtract(after,before)))
        self.command_path+=float(np.linalg.norm(np.subtract(command_after,command_before)))
        self.improving+=int(np.linalg.norm(np.subtract(goal,before))-np.linalg.norm(np.subtract(goal,after))>1e-8)
        if self.previous is not None:
            self.pairs+=1;self.reversed+=int(self.previous@action < -1e-9)
        self.previous=None if done else action.copy()

    def summary(self):
        return dict(steps=self.steps,actual_path_m=self.actual_path,command_path_m=self.command_path,
            mean_actual_displacement_m=self.actual_path/max(1,self.steps),
            actual_to_command_path_ratio=self.actual_path/self.command_path if self.command_path else None,
            toward_actual_fraction=self.actual_toward/self.actual_valid if self.actual_valid else None,
            toward_command_fraction=self.command_toward/self.command_valid if self.command_valid else None,
            distance_improving_fraction=self.improving/max(1,self.steps),
            reversal_fraction=self.reversed/self.pairs if self.pairs else None)
