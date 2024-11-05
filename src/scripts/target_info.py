from dataclasses import dataclass


@dataclass
class TargetInfo:
    name = "HD169142"
    plx = 8.7053e-3  # "

    @property
    def dist_pc(self):
        return 1 / self.plx
