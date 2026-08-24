import numpy as np
import matplotlib.pyplot as plt
from antenna import MovableAntenna, AntennaConfig
from irs import IRS, IRSConfig
from optimization import JointOptimizer


def run_simulation():
    """Run complete simulation of movable antenna with IRS"""
    
    print("="*60)
    print("Movable Antenna & IRS Simulation")
    print("="*60)
    
    # Antenna configuration
    antenna_config = AntennaConfig(
        frequency=2.4e9,  # 2.4 GHz
        wavelength=0.125,  # meters (c/f)
        gain=5.0,  # dBi
        beamwidth=30,  # degrees
        max_speed=5.0,  # m/s
        position=np.array([0, 0, 2.0])  # Starting position
    )
    
    # IRS configuration
    irs_config = IRSConfig(
        num_elements=16,  # 16-element linear array
        element_spacing=0.5,  # Half wavelength spacing
        position=np.array([5, 0, 2.0]),  # IRS position
        wavelength=antenna_config.wavelength,
        phase_resolution=3  # 3-bit phase quantization
    )
    
    # Create antenna and IRS
    antenna = MovableAntenna(antenna_config)
    irs = IRS(irs_config)
    
    print(f"\nAntenna Config:")
    print(f"  Frequency: {antenna_config.frequency/1e9:.1f} GHz")
    print(f"  Wavelength: {antenna_config.wavelength:.3f} m")
    print(f"  Starting Position: {antenna.position}")
    
    print(f"\nIRS Config:")
    print(f"  Number of Elements: {irs_config.num_elements}")
    print(f"  Position: {irs.position}")
    print(f"  Element Spacing: {irs_config.element_spacing} wavelengths")
    
    # Target position
    target_position = np.array([10, 0, 1.5])
    print(f"\nTarget Position: {target_position}")
    
    # Create optimizer
    optimizer = JointOptimizer(antenna, irs)
    
    # Run joint optimization
    print("\n" + "="*60)
    print("Starting Joint Optimization...")
    print("="*60)
    
    antenna_capacities, irs_gains = optimizer.joint_optimize(
        target_position, 
        num_iterations=30
    )
    
    # Plot results
    print("\n" + "="*60)
    print("Plotting Results...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Antenna capacity over iterations
    axes[0, 0].plot(antenna_capacities, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Capacity (bits/Hz)')
    axes[0, 0].set_title('Antenna Channel Capacity Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # IRS gain over iterations
    axes[0, 1].plot(irs_gains, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Reflection Gain (dB)')
    axes[0, 1].set_title('IRS Reflection Gain Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Phase shifts
    axes[1, 0].bar(range(len(irs.phase_shifts)), np.degrees(irs.phase_shifts))
    axes[1, 0].set_xlabel('Element Index')
    axes[1, 0].set_ylabel('Phase Shift (degrees)')
    axes[1, 0].set_title('IRS Element Phase Shifts')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 3D path visualization
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Plot positions
    ax3d.scatter(*antenna.position, color='blue', s=100, label='Antenna', marker='^')
    ax3d.scatter(*irs.position, color='red', s=100, label='IRS', marker='s')
    ax3d.scatter(*target_position, color='green', s=100, label='Target', marker='o')
    
    # Plot element positions
    element_positions = irs.element_positions
    ax3d.scatter(element_positions[:, 0], element_positions[:, 1], 
                element_positions[:, 2], color='red', alpha=0.5, s=30)
    
    # Draw paths
    ax3d.plot([antenna.position[0], target_position[0]], 
             [antenna.position[1], target_position[1]],
             [antenna.position[2], target_position[2]],
             'b--', alpha=0.5, label='Direct Path')
    
    ax3d.plot([antenna.position[0], irs.position[0]],
             [antenna.position[1], irs.position[1]],
             [antenna.position[2], irs.position[2]],
             'r--', alpha=0.5, label='TX to IRS')
    
    ax3d.plot([irs.position[0], target_position[0]],
             [irs.position[1], target_position[1]],
             [irs.position[2], target_position[2]],
             'g--', alpha=0.5, label='IRS to RX')
    
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('System Geometry')
    ax3d.legend()
    
    plt.tight_layout()
    plt.savefig('movable_antenna_irs_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'movable_antenna_irs_results.png'")
    plt.show()
    
    # Print final summary
    print("\n" + "="*60)
    print("Optimization Summary")
    print("="*60)
    print(f"Initial Capacity: {antenna_capacities[0]:.4f} bits/Hz")
    print(f"Final Capacity: {antenna_capacities[-1]:.4f} bits/Hz")
    print(f"Capacity Improvement: {(antenna_capacities[-1]/antenna_capacities[0] - 1)*100:.2f}%")
    print(f"\nInitial IRS Gain: {irs_gains[0]:.4f} dB")
    print(f"Final IRS Gain: {irs_gains[-1]:.4f} dB")
    print(f"Gain Improvement: {irs_gains[-1] - irs_gains[0]:.4f} dB")
    print(f"\nFinal Antenna Position: {antenna.position}")
    print(f"Target Position: {target_position}")
    print(f"Distance to Target: {np.linalg.norm(antenna.position - target_position):.4f} m")


if __name__ == "__main__":
    run_simulation()
