import numpy as np
from antenna import MovableAntenna, AntennaConfig
from irs import IRS, IRSConfig
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt


class JointOptimizer:
    """Joint optimization of movable antenna and IRS"""
    
    def __init__(self, antenna: MovableAntenna, irs: IRS):
        self.antenna = antenna
        self.irs = irs
        self.optimization_history = []
        
    def objective_function(self, antenna_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """Calculate channel capacity (simplified Shannon capacity)"""
        # Set antenna position
        self.antenna.position = antenna_pos.copy()
        
        # Get direct path gain
        direct_gain = self.antenna.get_channel_gain(target_pos)
        
        # Get reflected path gain through IRS
        reflected_gain = self.irs.get_reflection_gain(antenna_pos, target_pos)
        
        # Total received power (linear combination)
        direct_power = 10 ** (direct_gain / 10)
        reflected_power = 10 ** (reflected_gain / 10)
        total_power = direct_power + reflected_power
        
        # Assume noise power = 1
        snr = total_power / 1.0
        
        # Shannon capacity
        capacity = np.log2(1 + snr)
        
        return capacity
    
    def optimize_antenna_position(self, target_pos: np.ndarray, 
                                 num_iterations: int = 100,
                                 learning_rate: float = 0.1) -> List[float]:
        """Optimize antenna position using gradient ascent"""
        capacities = []
        
        for iteration in range(num_iterations):
            current_pos = self.antenna.position.copy()
            current_capacity = self.objective_function(current_pos, target_pos)
            capacities.append(current_capacity)
            
            # Compute gradient numerically
            perturbation = 0.01
            gradient = np.zeros(3)
            
            for dim in range(3):
                perturbed_pos = current_pos.copy()
                perturbed_pos[dim] += perturbation
                perturbed_capacity = self.objective_function(perturbed_pos, target_pos)
                
                gradient[dim] = (perturbed_capacity - current_capacity) / perturbation
            
            # Update antenna position
            new_pos = current_pos + learning_rate * gradient
            self.antenna.position = new_pos.copy()
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Capacity = {current_capacity:.4f} bits/Hz")
        
        return capacities
    
    def optimize_irs_phases(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
                          num_iterations: int = 50) -> List[float]:
        """Optimize IRS phase shifts"""
        gains = []
        
        for iteration in range(num_iterations):
            # Optimize phase shifts
            self.irs.optimize_phase_shifts(tx_pos - self.irs.position, 
                                          rx_pos - self.irs.position,
                                          num_iterations=10)
            
            # Calculate gain
            gain = self.irs.get_reflection_gain(tx_pos, rx_pos)
            gains.append(gain)
            
            if iteration % 10 == 0:
                print(f"IRS Optimization {iteration}: Gain = {gain:.4f} dB")
        
        return gains
    
    def joint_optimize(self, target_pos: np.ndarray,
                      num_iterations: int = 50) -> Tuple[List[float], List[float]]:
        """Jointly optimize antenna position and IRS phases"""
        antenna_capacities = []
        irs_gains = []
        
        for iteration in range(num_iterations):
            print(f"\n=== Joint Optimization Iteration {iteration} ===")
            
            # Optimize antenna position
            current_pos = self.antenna.position.copy()
            current_capacity = self.objective_function(current_pos, target_pos)
            
            perturbation = 0.01
            gradient = np.zeros(3)
            
            for dim in range(3):
                perturbed_pos = current_pos.copy()
                perturbed_pos[dim] += perturbation
                perturbed_capacity = self.objective_function(perturbed_pos, target_pos)
                gradient[dim] = (perturbed_capacity - current_capacity) / perturbation
            
            self.antenna.position = current_pos + 0.05 * gradient
            antenna_capacities.append(current_capacity)
            
            # Optimize IRS phases
            self.irs.optimize_phase_shifts(current_pos - self.irs.position,
                                          target_pos - self.irs.position,
                                          num_iterations=20)
            
            irs_gain = self.irs.get_reflection_gain(current_pos, target_pos)
            irs_gains.append(irs_gain)
            
            print(f"Antenna Capacity: {current_capacity:.4f} bits/Hz")
            print(f"IRS Gain: {irs_gain:.4f} dB")
            print(f"Antenna Position: {self.antenna.position}")
        
        return antenna_capacities, irs_gains
