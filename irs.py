import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class IRSConfig:
    """Configuration for Intelligent Reflecting Surface"""
    num_elements: int  # Number of reflecting elements
    element_spacing: float  # Wavelength units
    position: np.ndarray  # [x, y, z] position
    wavelength: float  # meters
    phase_resolution: int = 8  # Phase quantization bits


class IRS:
    """Intelligent Reflecting Surface with phase shift control"""
    
    def __init__(self, config: IRSConfig):
        self.config = config
        self.position = config.position.copy()
        self.phase_shifts = np.zeros(config.num_elements)  # Phase shifts in radians
        self.element_positions = self._calculate_element_positions()
        
    def _calculate_element_positions(self) -> np.ndarray:
        """Calculate positions of all IRS elements in a planar array"""
        # Assume linear array along x-axis
        num_elements = self.config.num_elements
        spacing = self.config.element_spacing * self.config.wavelength
        
        positions = np.zeros((num_elements, 3))
        for i in range(num_elements):
            positions[i, 0] = self.position[0] + (i - num_elements/2) * spacing
            positions[i, 1] = self.position[1]
            positions[i, 2] = self.position[2]
        
        return positions
    
    def set_phase_shifts(self, phase_shifts: np.ndarray):
        """Set phase shifts for all elements"""
        if len(phase_shifts) != self.config.num_elements:
            raise ValueError(f"Expected {self.config.num_elements} phase shifts")
        
        self.phase_shifts = phase_shifts % (2 * np.pi)
    
    def quantize_phase_shifts(self, phase_shifts: np.ndarray) -> np.ndarray:
        """Quantize phase shifts based on resolution"""
        num_levels = 2 ** self.config.phase_resolution
        quantized = np.round(phase_shifts / (2 * np.pi) * num_levels) * (2 * np.pi / num_levels)
        return quantized % (2 * np.pi)
    
    def compute_beamforming_vector(self, direction: np.ndarray) -> np.ndarray:
        """Compute beamforming vector for given direction"""
        # Normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Wavelength in meters
        wavelength = self.config.wavelength
        
        # Calculate phase difference for each element
        beamforming_vector = np.zeros(self.config.num_elements, dtype=complex)
        
        for i in range(self.config.num_elements):
            # Spatial signature
            element_pos = self.element_positions[i, :]
            path_diff = np.dot(element_pos, direction)
            phase = 2 * np.pi * path_diff / wavelength
            
            # Apply phase shift and create beamforming vector
            beamforming_vector[i] = np.exp(1j * (phase + self.phase_shifts[i]))
        
        return beamforming_vector / np.linalg.norm(beamforming_vector)
    
    def optimize_phase_shifts(self, tx_direction: np.ndarray, rx_direction: np.ndarray, 
                             num_iterations: int = 100) -> np.ndarray:
        """Optimize phase shifts using gradient ascent for channel gain"""
        wavelength = self.config.wavelength
        step_size = 0.1
        
        # Normalize directions
        tx_direction = tx_direction / (np.linalg.norm(tx_direction) + 1e-10)
        rx_direction = rx_direction / (np.linalg.norm(rx_direction) + 1e-10)
        
        for iteration in range(num_iterations):
            # Compute current channel gain
            tx_bf = self.compute_beamforming_vector(tx_direction)
            rx_bf = self.compute_beamforming_vector(rx_direction)
            
            # Reflected channel gain
            channel_response = np.abs(np.dot(rx_bf.conj(), tx_bf))
            
            # Gradient-based update
            for i in range(self.config.num_elements):
                # Compute phase gradient
                phase_perturb = 0.01
                phase_backup = self.phase_shifts[i]
                
                # Positive perturbation
                self.phase_shifts[i] = phase_backup + phase_perturb
                tx_bf_pos = self.compute_beamforming_vector(tx_direction)
                rx_bf_pos = self.compute_beamforming_vector(rx_direction)
                gain_pos = np.abs(np.dot(rx_bf_pos.conj(), tx_bf_pos))
                
                # Negative perturbation
                self.phase_shifts[i] = phase_backup - phase_perturb
                tx_bf_neg = self.compute_beamforming_vector(tx_direction)
                rx_bf_neg = self.compute_beamforming_vector(rx_direction)
                gain_neg = np.abs(np.dot(rx_bf_neg.conj(), tx_bf_neg))
                
                # Gradient estimate
                gradient = (gain_pos - gain_neg) / (2 * phase_perturb)
                self.phase_shifts[i] = phase_backup + step_size * gradient
                self.phase_shifts[i] = self.phase_shifts[i] % (2 * np.pi)
        
        return self.phase_shifts
    
    def get_reflection_gain(self, tx_position: np.ndarray, rx_position: np.ndarray) -> float:
        """Calculate reflection gain from transmitter to receiver"""
        # Direction from IRS to transmitter
        tx_dir = tx_position - self.position
        tx_dir = tx_dir / (np.linalg.norm(tx_dir) + 1e-10)
        
        # Direction from IRS to receiver
        rx_dir = rx_position - self.position
        rx_dir = rx_dir / (np.linalg.norm(rx_dir) + 1e-10)
        
        # Compute beamforming vectors
        tx_bf = self.compute_beamforming_vector(tx_dir)
        rx_bf = self.compute_beamforming_vector(rx_dir)
        
        # Channel gain magnitude
        channel_gain = np.abs(np.dot(rx_bf.conj(), tx_bf))
        
        # Convert to dB
        gain_db = 20 * np.log10(channel_gain + 1e-10)
        
        return gain_db
