from dataclasses import dataclass


@dataclass
class TargetInfo:
    name = "HD169142"
    plx = 8.7053e-3  # " +- 0.0268e-3
    inclination = 12.5 # deg
    pos_angle = 5 # deg, location of far side minor axis
    stellar_mass = 1.65 # Msun

    @property
    def dist_pc(self):
        return 1 / self.plx

target_info = TargetInfo()
