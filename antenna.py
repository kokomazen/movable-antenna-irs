import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class AntennaConfig:
    """Configuration for movable antenna system"""
    frequency: float  # Hz
    wavelength: float  # meters
    gain: float  # dBi
    beamwidth: float  # degrees
    max_speed: float  # m/s
    position: np.ndarray  # [x, y, z] coordinates


class MovableAntenna:
    """Movable antenna with position and orientation control"""
    
    def __init__(self, config: AntennaConfig):
        self.config = config
        self.position = config.position.copy()
        self.orientation = np.array([0, 0, 0])  # yaw, pitch, roll in radians
        self.velocity = np.array([0, 0, 0])
        
    def move_to_position(self, target_position: np.ndarray, time_step: float = 0.1):
        """Move antenna towards target position"""
        direction = target_position - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return distance
        
        # Calculate velocity towards target with speed limit
        direction_normalized = direction / distance
        max_displacement = self.config.max_speed * time_step
        
        if distance <= max_displacement:
            self.position = target_position.copy()
        else:
            self.position += direction_normalized * max_displacement
        
        return np.linalg.norm(target_position - self.position)
    
    def set_orientation(self, yaw: float, pitch: float, roll: float = 0):
        """Set antenna orientation (in radians)"""
        self.orientation = np.array([yaw, pitch, roll])
    
    def get_radiation_pattern(self, direction: np.ndarray, frequency: float = None) -> float:
        """Calculate antenna gain in given direction"""
        if frequency is None:
            frequency = self.config.frequency
        
        # Normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Calculate beam center
        yaw, pitch, _ = self.orientation
        beam_center = np.array([
            np.sin(yaw) * np.cos(pitch),
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch)
        ])
        
        # Calculate angle between beam center and target direction
        cos_angle = np.dot(beam_center, direction)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angle_deg = np.degrees(angle)
        
        # Cosine pattern with beamwidth
        half_beamwidth = self.config.beamwidth / 2
        if angle_deg <= half_beamwidth:
            gain = self.config.gain * np.cos(angle)
        else:
            gain = self.config.gain * np.cos(angle) * np.exp(-((angle_deg - half_beamwidth) / 10)**2)
        
        return max(gain, -30)  # Minimum gain floor
    
    def calculate_path_loss(self, target_position: np.ndarray) -> float:
        """Calculate path loss using Friis equation"""
        distance = np.linalg.norm(target_position - self.position)
        
        if distance < 0.1:
            distance = 0.1
        
        # Friis equation: PL = 20*log10(4*pi*d/lambda)
        wavelength = self.config.wavelength
        path_loss = 20 * np.log10((4 * np.pi * distance) / wavelength)
        
        return path_loss
    
    def get_channel_gain(self, target_position: np.ndarray) -> float:
        """Calculate total channel gain including antenna gain and path loss"""
        direction = target_position - self.position
        antenna_gain = self.get_radiation_pattern(direction)
        path_loss = self.calculate_path_loss(target_position)
        
        channel_gain = antenna_gain - path_loss
        return channel_gain
