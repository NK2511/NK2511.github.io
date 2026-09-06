import numpy as np
import plotly.graph_objects as go
import random
from pyswarm import pso
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from typing import List,Tuple,Optional
import matplotlib.pyplot as plt
import time

def calculate_end_effector_position(link_vectors: List[List[float]], axes: List[List[float]], thetas: List[float]):
    """Calculates the end effector position for n links."""
    current_position = np.array([0.0, 0.0, 0.0])
    current_rotation = R.identity()
    positions = [current_position.copy()]

    for i in range(len(link_vectors)):
        if np.allclose(axes[i], [0, 0, 0]):
            rotation_matrix = R.identity()
        else:
            rotation_matrix = R.from_rotvec(np.array(axes[i]) * thetas[i])
        current_rotation = current_rotation * rotation_matrix
        current_position += current_rotation.apply(np.array(link_vectors[i]))
        positions.append(current_position.copy())
    return positions

def generate_workspace_points(link_vectors: List[List[float]], axes: List[List[float]], num_points: int = 10000):
    """Generate workspace points for n links."""
    points = []
    num_links = len(link_vectors)
    for _ in range(num_points):
        thetas = [random.uniform(0, 2 * np.pi) for _ in range(num_links)]
        end_effector_positions = calculate_end_effector_position(link_vectors, axes, thetas)
        points.extend(end_effector_positions)
    return np.array(points)

def plot_plane(origin: List[float], normal: List[float], size: float = 1.0):
    """Plot a plane given an origin and a normal vector."""
    d = size / 2
    normal = np.array(normal, dtype=float)

    if not np.allclose(normal, [0, 0, 0]):
        normal = normal / np.linalg.norm(normal)
        if np.allclose(normal, [1, 0, 0]):
            v1, v2 = np.array([0, 1, 0]), np.array([0, 0, 1])
        elif np.allclose(normal, [0, 1, 0]):
            v1, v2 = np.array([1, 0, 0]), np.array([0, 0, 1])
        elif np.allclose(normal, [0, 0, 1]):
            v1, v2 = np.array([1, 0, 0]), np.array([0, 1, 0])
        else:
            v1 = np.cross(normal, [1, 0, 0])
            if np.allclose(v1, [0, 0, 0]):
                v1 = np.cross(normal, [0, 1, 0])
            v1 /= np.linalg.norm(v1)
            v2 = np.cross(normal, v1)

        plane_points = np.array([
            origin + d * (v1 + v2),
            origin + d * (v1 - v2),
            origin + d * (-v1 - v2),
            origin + d * (-v1 + v2),
        ])
    else:
        plane_points = np.array([
            [origin[0] - d, origin[1] - d, origin[2]],
            [origin[0] + d, origin[1] - d, origin[2]],
            [origin[0] + d, origin[1] + d, origin[2]],
            [origin[0] - d, origin[1] + d, origin[2]]
        ])
    return plane_points

def plot_robot_arm_pose_with_workspace(link_vectors: List[List[float]], axes: List[List[float]], thetas: List[float]):
    """Plot the robot arm pose and its workspace."""
    positions = calculate_end_effector_position(link_vectors, axes, thetas)
    workspace_points = generate_workspace_points(link_vectors, axes)

    fig = go.Figure()

    # Workspace cloud
    fig.add_trace(go.Scatter3d(
        x=workspace_points[:, 0],
        y=workspace_points[:, 1],
        z=workspace_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='black', opacity=0.7),
        name='Workspace'
    ))

    # Joints and links
    for i in range(1, len(positions)):
        # Link line
        fig.add_trace(go.Scatter3d(
            x=[positions[i - 1][0], positions[i][0]],
            y=[positions[i - 1][1], positions[i][1]],
            z=[positions[i - 1][2], positions[i][2]],
            mode='lines',
            line=dict(color=['red', 'orange', 'green', 'purple', 'cyan'][i % 5], width=7),
            name=f'Link {i}'
        ))

        # Joint marker
        fig.add_trace(go.Scatter3d(
            x=[positions[i][0]],
            y=[positions[i][1]],
            z=[positions[i][2]],
            mode='markers',
            marker=dict(size=5, color='purple'),
            name=f'Joint {i}'
        ))

    # End effector marker
    fig.add_trace(go.Scatter3d(
        x=[positions[-1][0]],
        y=[positions[-1][1]],
        z=[positions[-1][2]],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='End Effector'
    ))

    # Axis planes
    for i, axis in enumerate(axes):
        origin = positions[i]
        axis_end = origin + np.array(axis) * 0.5
        fig.add_trace(go.Scatter3d(
            x=[origin[0], axis_end[0]],
            y=[origin[1], axis_end[1]],
            z=[origin[2], axis_end[2]],
            mode='lines',
            line=dict(color='black', width=3),
            name=f'Axis {i + 1}'
        ))

        # Plane
        plane_points = plot_plane(origin, axis)
        fig.add_trace(go.Mesh3d(
            x=plane_points[:, 0],
            y=plane_points[:, 1],
            z=plane_points[:, 2],
            opacity=0.3,
            color='lightgray',
            i=[0, 1, 2, 2, 3, 0],
            j=[1, 2, 3, 3, 0, 1],
            k=[2, 3, 0, 0, 1, 2],
            name=f'Plane {i + 1}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-2, 2], autorange=False),
            yaxis=dict(title='Y', range=[-2, 2], autorange=False),
            zaxis=dict(title='Z', range=[-1, 3], autorange=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title='Animated Robot Arm Path'
    )
    fig.show()

def objective_function(thetas: List[float], link_vectors: List[List[float]], axes: List[List[float]], target_position: List[float]) -> float:
    """Objective function for PSO: distance to target."""
    positions = calculate_end_effector_position(link_vectors, axes, thetas)
    end_effector = positions[-1]
    return np.linalg.norm(np.array(target_position) - np.array(end_effector))

def hybrid_path(target_path: List[List[float]], link_vectors: List[List[float]], axes: List[List[float]]) -> List[List[float]]:

    joint_angles_path = []
    num_joints = len(link_vectors)
    lb = [-2 * np.pi] * num_joints
    ub = [2 * np.pi] * num_joints
    pso_options = {'swarmsize': 50, 'maxiter': 200}
    local_opt_options = {'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
    max_error = 0.01  # Maximum allowed error (tunable)
    max_retries = 5    # Maximum number of PSO retries

    first_thetas = None
    retries = 0
    while retries < max_retries:
        print(f"Solving for the first point using PSO (attempt {retries + 1})...")
        first_thetas, _ = pso(
            objective_function, lb, ub,
            args=(link_vectors, axes, target_path[0]),
            **pso_options
        )
        ee_pos_first = calculate_end_effector_position(link_vectors, axes, first_thetas.tolist())[-1]
        first_error = np.linalg.norm(np.array(target_path[0]) - np.array(ee_pos_first))
        if first_error <= max_error:
            print("PSO succeeded for the first point.")
            joint_angles_path.append(first_thetas.tolist())
            break
        else:
            print(f"PSO failed for the first point. Error: {first_error:.4f} > {max_error:.4f}. Retrying...")
            retries += 1

    if first_thetas is None:
        raise RuntimeError("PSO failed to find a valid solution for the first point after multiple retries.")

    # 2. Use local optimization (L-BFGS-B) for the rest of the points
    for i in range(1, len(target_path)):
        print(f"Solving for point {i + 1} using local optimization...")
        previous_target = np.array(target_path[i - 1])
        current_target = np.array(target_path[i])
        target_delta = current_target - previous_target

        initial_guess = joint_angles_path[-1]

        if np.linalg.norm(target_delta) > 1e-6 and len(joint_angles_path) > 0:
            current_ee_positions_prev_joints = calculate_end_effector_position(link_vectors, axes, joint_angles_path[-1])
            current_ee_pos_prev_joints = current_ee_positions_prev_joints[-1]
            if np.linalg.norm(current_ee_pos_prev_joints - previous_target) > 1e-6:
                jacobian_approx = np.random.rand(3, num_joints) - 0.5
                try:
                    joint_delta_approx = np.linalg.pinv(jacobian_approx) @ target_delta * 0.1
                    initial_guess = np.array(joint_angles_path[-1]) + joint_delta_approx
                    initial_guess = np.clip(initial_guess, lb, ub).tolist()
                except np.linalg.LinAlgError:
                    initial_guess = joint_angles_path[-1]

        res = minimize(
            objective_function,
            initial_guess,
            args=(link_vectors, axes, target_path[i]),
            bounds=[(lb[j], ub[j]) for j in range(num_joints)],
            method='L-BFGS-B',
            options=local_opt_options
        )
        ee_pos = calculate_end_effector_position(link_vectors, axes, res.x.tolist())[-1]
        error = np.linalg.norm(np.array(target_path[i]) - np.array(ee_pos))

        if error > max_error:
            print(f"Local optimization error too high at point {i+1}: {error}. Restarting from PSO")
            return hybrid_path(target_path, link_vectors, axes)  # Restart IK calculation
        joint_angles_path.append(res.x.tolist())
    return joint_angles_path

def animate_robot(link_vectors: List[List[float]],
                  axes: List[List[float]],
                  theta_sequence: List[List[float]],
                  obstacles: Optional[List[dict]] = None):

    frames = []
    x_vals, y_vals, z_vals = [], [], []
    end_effector_trace_x, end_effector_trace_y, end_effector_trace_z = [], [], []

    # Calculate the overall range for each axis
    for thetas in theta_sequence:  # Iterate *once* to get overall range
        positions = calculate_end_effector_position(link_vectors, axes, thetas)
        x, y, z = zip(*positions)
        x_vals.extend(x)
        y_vals.extend(y)
        z_vals.extend(z)
    x_min, x_max = min(x_vals) - 0.5, max(x_vals) + 0.5
    y_min, y_max = min(y_vals) - 0.5, max(y_vals) + 0.5
    z_min, z_max = min(z_vals) - 0.5, max(z_vals) + 0.5

    # Determine the maximum range to enforce a cube aspect ratio
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    x_range = [x_mid - max_range / 2, x_mid + max_range / 2]
    y_range = [y_mid - max_range / 2, y_mid + max_range / 2]
    z_range = [z_mid - max_range / 2, z_mid + max_range / 2]

    # Initial figure
    initial_positions = calculate_end_effector_position(link_vectors, axes, theta_sequence[0])
    x0, y0, z0 = zip(*initial_positions)
    initial_end_effector = initial_positions[-1]

    data = [
        go.Scatter3d(
            x=x0, y=y0, z=z0,
            mode='lines+markers',
            line=dict(color='blue', width=6),
            marker=dict(size=4, color='red'),
            showlegend=False,
            name='Robot Arm'
        ),
        go.Scatter3d(
            x=[initial_end_effector[0]],
            y=[initial_end_effector[1]],
            z=[initial_end_effector[2]],
            mode='markers',
            marker=dict(size=8, color='green'),
            name='End Effector'
        ),
        go.Scatter3d(
            x=end_effector_trace_x[:1],
            y=end_effector_trace_y[:1],
            z=end_effector_trace_z[:1],
            mode='lines',
            line=dict(color='green', width=3, dash='dash'),
            name='End Effector Trace'
        ),
    ]

    # Add obstacles to the initial plot
    if obstacles:
        for i, obs in enumerate(obstacles):
            if obs['type'] == 'sphere':
                # This is a simplified representation for a sphere in plotly
                # A mesh would be more accurate but this is faster for visualization
                data.append(go.Scatter3d(
                    x=[obs['center'][0]], y=[obs['center'][1]], z=[obs['center'][2]],
                    mode='markers',
                    marker=dict(size=obs['radius']*35, color='grey', opacity=0.5), # Size is heuristic
                    name=f'Obstacle {i+1}'
                ))


    plane_traces = []
    axis_traces = []
    for i, axis in enumerate(axes):
        origin = initial_positions[i]
        rotation_matrix = R.identity()
        for j in range(i):
            if not np.allclose(axes[j], [0, 0, 0]):
                rotation_matrix = rotation_matrix * R.from_rotvec(
                    np.array(axes[j]) * theta_sequence[0][j])
        rotated_axis = rotation_matrix.apply(np.array(axis))
        axis_end = origin + rotated_axis * 0.5
        axis_trace = go.Scatter3d(
            x=[origin[0], axis_end[0]],
            y=[origin[1], axis_end[1]],
            z=[origin[2], axis_end[2]],
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False,
        )
        data.append(axis_trace)
        axis_traces.append(axis_trace)
        plane_points = plot_plane(origin, rotated_axis)
        plane_trace = go.Mesh3d(
            x=plane_points[:, 0],
            y=plane_points[:, 1],
            z=plane_points[:, 2],
            opacity=0.3,
            color='lightgray',
            i=[0, 1, 2, 2, 3, 0],
            j=[1, 2, 3, 3, 0, 1],
            k=[2, 3, 0, 0, 1, 2],
            showlegend=False
        )
        data.append(plane_trace)
        plane_traces.append(plane_trace)

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title="Robot Arm Animation",
            scene=dict(
                xaxis=dict(title='X', range=x_range),
                yaxis=dict(title='Y', range=y_range),
                zaxis=dict(title='Z', range=z_range),
                aspectmode='cube'
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 0}}]
                )]
            )]
        ),
    )

    for k, thetas in enumerate(theta_sequence):
        positions = calculate_end_effector_position(link_vectors, axes, thetas)
        x, y, z = zip(*positions)
        end_effector = positions[-1]
        end_effector_trace_x.append(end_effector[0])
        end_effector_trace_y.append(end_effector[1])
        end_effector_trace_z.append(end_effector[2])
        frame_data = [
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                line=dict(color='blue', width=6),
                marker=dict(size=4, color='red'),
                showlegend=False,
            ),
            go.Scatter3d(
                x=end_effector_trace_x[:k + 1],
                y=end_effector_trace_y[:k + 1],
                z=end_effector_trace_z[:k + 1],
                mode='lines',
                line=dict(color='green', width=3, dash='dash'),
                showlegend=False
            ),
            go.Scatter3d(
                x=[end_effector[0]],
                y=[end_effector[1]],
                z=[end_effector[2]],
                mode='markers',
                marker=dict(size=8, color='green'),
                showlegend=False
            )
        ]
        for i, axis in enumerate(axes):
            origin = positions[i]
            rotation_matrix = R.identity()
            for j in range(i):
                if not np.allclose(axes[j], [0, 0, 0]):
                    rotation_matrix = rotation_matrix * R.from_rotvec(
                        np.array(axes[j]) * thetas[j])
            rotated_axis = rotation_matrix.apply(np.array(axis))
            axis_end = origin + rotated_axis * 0.5
            frame_data.append(go.Scatter3d(
                x=[origin[0], axis_end[0]],
                y=[origin[1], axis_end[1]],
                z=[origin[2], axis_end[2]],
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=False,
            ))
            plane_points = plot_plane(origin, rotated_axis)
            frame_data.append(go.Mesh3d(
                x=plane_points[:, 0],
                y=plane_points[:, 1],
                z=plane_points[:, 2],
                opacity=0.3,
                color='lightgray',
                i=[0, 1, 2, 2, 3, 0],
                j=[1, 2, 3, 3, 0, 1],
                k=[2, 3, 0, 0, 1, 2],
                showlegend=False
            ))
        frames.append(go.Frame(data=frame_data, name=str(k)))

    # Add obstacles to the layout so they are static and don't disappear
    if obstacles:
        for i, obs in enumerate(obstacles):
            if obs['type'] == 'sphere':
                # We add a static trace for the obstacle
                fig.add_trace(go.Scatter3d(
                    x=[obs['center'][0]], y=[obs['center'][1]], z=[obs['center'][2]],
                    mode='markers',
                    marker=dict(size=obs['radius']*35, color='rgba(128, 128, 128, 0.5)'),
                    name=f'Obstacle {i+1}',
                    hoverinfo='skip'
                ))

    fig.frames = frames
    fig.show()

def plot_joint_angles(joint_angles: List[List[float]], target_path: List[List[float]], link_vectors: List[List[float]], axes: List[List[float]], title: str):
    joint_angles = np.array(joint_angles)
    num_points, num_joints = joint_angles.shape

    # Compute errors at each point
    errors = []
    for i in range(num_points):
        ee_pos = calculate_end_effector_position(link_vectors, axes, joint_angles[i].tolist())[-1]
        error = np.linalg.norm(np.array(target_path[i]) - np.array(ee_pos))
        errors.append(error)

    plt.figure(figsize=(12, 6))

    # Plot joint angles
    for j in range(num_joints):
        plt.plot(joint_angles[:, j], label=f'Joint {j+1}', linewidth=1)

    # Plot error in thick red
    plt.plot(errors, color='red', linewidth=3, label='End-effector error')

    plt.title(f"Joint Angles and Error - {title}")
    plt.xlabel("Time step")
    plt.ylabel("Angle (radians) / Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def accuracy_graph(joint_angles_list, target_path, link_vectors, axes):
    errors = []
    for angles, target in zip(joint_angles_list, target_path):
        positions = calculate_end_effector_position(link_vectors, axes, angles)
        end_effector_pos = positions[-1]
        error = np.linalg.norm(np.array(end_effector_pos) - np.array(target))
        errors.append(error)
    print(errors)
    return errors

def plot_joint_angles_on_ax(ax, joint_angles):
    joint_angles = np.array(joint_angles).T
    for i, joint in enumerate(joint_angles):
        ax.plot(joint, label=f'Joint {i+1}')
    ax.legend()

def generate_random_loop_path(points: int = 400, seed: int = None) -> List[List[float]]:

    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 2 * np.pi, points)
    
    def smooth_component():
        result = np.zeros_like(t)
        for _ in range(np.random.randint(2900, 3000)):  # 2 to 4 sine components
            freq = np.random.randint(1, 4)
            amp = np.random.uniform(0.2, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            result += amp * np.sin(freq * t + phase)
        return result

    x = smooth_component()
    y = smooth_component()
    z = smooth_component()

    # Normalize the path to fit within a cube [-1, 1]
    x /= np.max(np.abs(x))
    y /= np.max(np.abs(y))
    z /= np.max(np.abs(z))

    return np.stack([x, y, z], axis=1).tolist()

def fast_hybrid_path2(target_path, link_vectors, axes, angle_limits=None):
    import numpy as np
    from scipy.optimize import minimize
    from pyswarm import pso

    max_total_attempts = 5
    max_pso_retries = 5
    max_error = 0.01

    if angle_limits is None:
        angle_limits = [(-2 * np.pi, 2 * np.pi)] * len(link_vectors)

    num_joints = len(link_vectors)
    lb = [limit[0] for limit in angle_limits]
    ub = [limit[1] for limit in angle_limits]
    bounds = list(zip(lb, ub))

    pso_options = {'swarmsize': 300, 'maxiter': 100}
    local_opt_options = {'maxiter': 100, 'ftol': 1e-4, 'gtol': 1e-4}

    best_run_length = 0
    best_run_path = []
    overall_best_path = []

    # --- Helper function to solve a segment ---
    def solve_segment_hybrid(segment_targets, initial_guess):
        joint_path = [initial_guess]
        first_target = segment_targets[0]

        # Initial PSO
        for _ in range(max_pso_retries):
            first_thetas, _ = pso(
                objective_function, lb, ub,
                args=(link_vectors, axes, first_target),
                **pso_options
            )
            ee = calculate_end_effector_position(link_vectors, axes, first_thetas)[-1]
            if np.linalg.norm(np.array(first_target) - np.array(ee)) <= max_error:
                joint_path.append(first_thetas)
                break
        else:
            return None

        for i in range(1, len(segment_targets)):
            target = segment_targets[i]
            prev_joint = joint_path[-1]

            prev_target = np.array(segment_targets[i - 1])
            delta = np.array(target) - prev_target
            guess = prev_joint

            # Quick adjustment if needed
            prev_ee = calculate_end_effector_position(link_vectors, axes, prev_joint)[-1]
            if np.linalg.norm(prev_ee - prev_target) > 1e-6:
                jac_approx = np.random.rand(3, num_joints) - 0.5
                try:
                    adj = np.linalg.pinv(jac_approx) @ delta * 0.1
                    guess = np.clip(np.array(prev_joint) + adj, lb, ub)
                except np.linalg.LinAlgError:
                    pass

            res = minimize(
                objective_function, guess,
                args=(link_vectors, axes, target),
                bounds=bounds,
                method='L-BFGS-B',
                options=local_opt_options
            )
            ee = calculate_end_effector_position(link_vectors, axes, res.x)[-1]
            if np.linalg.norm(np.array(target) - np.array(ee)) > max_error:
                return None
            joint_path.append(res.x.tolist())
        return joint_path

    # --- Main Loop ---
    for attempt in range(max_total_attempts):
        np.random.seed(attempt)
        joint_path = []

        # Initial point with PSO
        for _ in range(max_pso_retries):
            thetas, _ = pso(
                objective_function, lb, ub,
                args=(link_vectors, axes, target_path[0]),
                **pso_options
            )
            ee = calculate_end_effector_position(link_vectors, axes, thetas)[-1]
            if np.linalg.norm(np.array(target_path[0]) - np.array(ee)) <= max_error:
                joint_path.append(thetas)
                break
        else:
            continue

        success = True
        for i in range(1, len(target_path)):
            prev_joint = joint_path[-1]
            prev_target = np.array(target_path[i - 1])
            target = np.array(target_path[i])
            delta = target - prev_target
            guess = prev_joint

            # Quick adjustment
            prev_ee = calculate_end_effector_position(link_vectors, axes, prev_joint)[-1]
            if np.linalg.norm(prev_ee - prev_target) > 1e-6:
                jac_approx = np.random.rand(3, num_joints) - 0.5
                try:
                    adj = np.linalg.pinv(jac_approx) @ delta * 0.1
                    guess = np.clip(np.array(prev_joint) + adj, lb, ub)
                except np.linalg.LinAlgError:
                    pass

            res = minimize(
                objective_function, guess,
                args=(link_vectors, axes, target),
                bounds=bounds,
                method='L-BFGS-B',
                options=local_opt_options
            )
            ee = calculate_end_effector_position(link_vectors, axes, res.x)[-1]
            if np.linalg.norm(target - np.array(ee)) > max_error:
                success = False
                break
            joint_path.append(res.x.tolist())

        if success:
            return joint_path

        if len(joint_path) > best_run_length:
            best_run_length = len(joint_path)
            best_run_path = joint_path.copy()
            overall_best_path = joint_path.copy()

    # --- Try hybrid rescue ---
    if best_run_path:
        remaining_targets = target_path[len(best_run_path):]
        for _ in range(max_total_attempts):
            seg = solve_segment_hybrid(remaining_targets, best_run_path[-1])
            if seg:
                overall_best_path.extend(seg[1:])
                return overall_best_path
        return overall_best_path

    raise RuntimeError("Failed to solve any portion of the path.")

def plot_workspace_with_path(link_vectors: List[List[float]],
                            axes: List[List[float]],
                            path: List[List[float]],
                            title: str = "Robot Arm Workspace with Path") -> None:

    # Ensure path is a numpy array for easier handling
    path = np.array(path)

    positions = [[0,0,0]] #start position

    workspace_points = generate_workspace_points(link_vectors, axes)

    fig = go.Figure()

    # Workspace cloud
    fig.add_trace(go.Scatter3d(
        x=workspace_points[:, 0],
        y=workspace_points[:, 1],
        z=workspace_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='black', opacity=0.7),
        name='Workspace'
    ))



    # Desired Path
    fig.add_trace(go.Scatter3d(
        x=path[:, 0],
        y=path[:, 1],
        z=path[:, 2],
        mode='lines', # Changed to 'lines'
        line=dict(color='blue', width=5),  # Increased width, removed dash
        name='Desired Path'
    ))



    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X',  autorange=True),  # Let the range be determined automatically
            yaxis=dict(title='Y',  autorange=True),
            zaxis=dict(title='Z',  autorange=True),
            aspectmode='auto', # Important: Use 'auto' to adjust aspect ratio
            # aspectratio=dict(x=1, y=1, z=1), # Remove this
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title=title
    )
    fig.show()

def multi_hybrid_paths(target_path, link_vectors, axes, num_paths=5, angle_limits=None):
    import numpy as np
    from scipy.optimize import minimize
    from pyswarm import pso

    max_pso_retries = 5
    max_error = 0.01

    if angle_limits is None:
        angle_limits = [(-2 * np.pi, 2 * np.pi)] * len(link_vectors)

    num_joints = len(link_vectors)
    lb = [limit[0] for limit in angle_limits]
    ub = [limit[1] for limit in angle_limits]
    bounds = list(zip(lb, ub))

    pso_options = {'swarmsize': 300, 'maxiter': 100}
    local_opt_options = {'maxiter': 100, 'ftol': 1e-4, 'gtol': 1e-4}

    all_paths = []

    def is_unique_path(new_path):
        for path in all_paths:
            diffs = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(path, new_path)]
            if all(d < 1e-2 for d in diffs):
                return False
        return True

    while len(all_paths) < num_paths:
        joint_path = []

        # Initial point with PSO
        for _ in range(max_pso_retries):
            thetas, _ = pso(
                objective_function, lb, ub,
                args=(link_vectors, axes, target_path[0]),
                **pso_options
            )
            ee = calculate_end_effector_position(link_vectors, axes, thetas)[-1]
            if np.linalg.norm(np.array(target_path[0]) - np.array(ee)) <= max_error:
                joint_path.append(thetas.tolist())
                break
        else:
            continue  # Retry a new path

        success = True
        for i in range(1, len(target_path)):
            prev_joint = joint_path[-1]
            prev_target = np.array(target_path[i - 1])
            target = np.array(target_path[i])
            delta = target - prev_target
            guess = prev_joint

            # Quick adjustment
            prev_ee = calculate_end_effector_position(link_vectors, axes, prev_joint)[-1]
            if np.linalg.norm(prev_ee - prev_target) > 1e-6:
                jac_approx = np.random.rand(3, num_joints) - 0.5
                try:
                    adj = np.linalg.pinv(jac_approx) @ delta * 0.1
                    guess = np.clip(np.array(prev_joint) + adj, lb, ub)
                except np.linalg.LinAlgError:
                    pass

            res = minimize(
                objective_function, guess,
                args=(link_vectors, axes, target),
                bounds=bounds,
                method='L-BFGS-B',
                options=local_opt_options
            )
            ee = calculate_end_effector_position(link_vectors, axes, res.x)[-1]
            if np.linalg.norm(target - np.array(ee)) > max_error:
                success = False
                break
            joint_path.append(res.x.tolist())

        if success and is_unique_path(joint_path):
            all_paths.append(joint_path)

    return all_paths

def compute_jacobian(link_vectors: List[List[float]], axes: List[List[float]], thetas: List[float]) -> np.ndarray:
    num_joints = len(link_vectors)
    J = np.zeros((3, num_joints))
    positions = calculate_end_effector_position(link_vectors, axes, thetas)
    end_effector_pos = positions[-1]
    rotation = R.identity()

    for i in range(num_joints):
        z = rotation.apply(axes[i])
        p_i = positions[i]
        J[:, i] = np.cross(z, end_effector_pos - p_i)
        rotation *= R.from_rotvec(np.array(axes[i]) * thetas[i])

    return J

def standard_jacobian_path(
    target_path: List[List[float]],
    link_vectors: List[List[float]],
    axes: List[List[float]],
    max_iterations: int = 100,
    damping: float = 0.01,
    tolerance: float = 1e-6
) -> List[List[float]]:
    num_joints = len(link_vectors)
    joint_angles_path = []
    current_angles = np.zeros(num_joints)

    for target in target_path:
        for _ in range(max_iterations):
            positions = calculate_end_effector_position(link_vectors, axes, current_angles)
            current_pos = positions[-1]
            error = np.array(target) - current_pos
            if np.linalg.norm(error) < tolerance:
                break

            J = compute_jacobian(link_vectors, axes, current_angles)
            JT = J.T
            JJt = J @ JT
            dq = JT @ np.linalg.inv(JJt + (damping ** 2) * np.eye(3)) @ error

            current_angles += dq  # Basic update (no adaptive step)

        joint_angles_path.append(current_angles.copy())

    return joint_angles_path

def generate_non_linear_path(points=100):
    t = np.linspace(0, 2*np.pi, points)
    x = np.sin(t) + 0.5 * np.sin(3 * t)
    y = np.cos(t) + 0.3 * np.cos(2 * t)
    z = 0.5 * np.sin(5 * t)
    return np.stack([x, y, z], axis=1)

def fast_hybrid_path2_optimized(target_path, link_vectors, axes, angle_limits=None):
    import numpy as np
    from scipy.optimize import minimize
    from pyswarm import pso # Keep pso for initial seed if needed

    max_total_attempts = 5
    max_pso_retries = 5
    max_error = 0.01

    if angle_limits is None:
        angle_limits = [(-2 * np.pi, 2 * np.pi)] * len(link_vectors)

    num_joints = len(link_vectors)
    lb = [limit[0] for limit in angle_limits]
    ub = [limit[1] for limit in angle_limits]
    bounds = list(zip(lb, ub))

    pso_options = {'swarmsize': 300, 'maxiter': 100} # You might reduce this
    # Use a method that can use the Jacobian if available
    local_opt_options = {'maxiter': 100, 'ftol': 1e-4, 'gtol': 1e-4}

    # ... (rest of your initial setup for best_run_length, etc.)

    overall_best_path = []

    # --- Helper function to solve a segment (modified to use Jacobian) ---
    def solve_segment_hybrid(segment_targets, initial_guess):
        joint_path = [initial_guess]
        # Skip initial PSO for segments, rely on local opt from previous point
        # unless it consistently fails.

        for i in range(len(segment_targets)):
            target = segment_targets[i]
            prev_joint = joint_path[-1] if joint_path else initial_guess
            guess = prev_joint

            res = minimize(
                objective_function, guess,
                args=(link_vectors, axes, target),
                bounds=bounds,
                method='SLSQP', # Consider SLSQP or trust-constr with jac=calculate_jacobian
                # jac=calculate_jacobian, # Uncomment if you have a proper analytical Jacobian
                options=local_opt_options
            )
            ee = calculate_end_effector_position(link_vectors, axes, res.x)[-1]
            if np.linalg.norm(np.array(target) - np.array(ee)) > max_error:
                # If local opt fails, try PSO for this specific point
                # (This is where your existing hybrid logic for segments could fit)
                for _ in range(max_pso_retries):
                    thetas_pso, _ = pso(
                        objective_function, lb, ub,
                        args=(link_vectors, axes, target),
                        **pso_options
                    )
                    ee_pso = calculate_end_effector_position(link_vectors, axes, thetas_pso)[-1]
                    if np.linalg.norm(np.array(target) - np.array(ee_pso)) <= max_error:
                        joint_path.append(thetas_pso.tolist())
                        break
                else:
                    return None # Still failed after PSO
            else:
                joint_path.append(res.x.tolist())
        return joint_path

    # --- Main Loop (modified to use Jacobian and prioritize local opt) ---
    for attempt in range(max_total_attempts):
        np.random.seed(attempt)
        joint_path = []

        # Initial point with PSO (keep this, it's a good global seed)
        initial_thetas_found = False
        for _ in range(max_pso_retries):
            thetas, _ = pso(
                objective_function, lb, ub,
                args=(link_vectors, axes, target_path[0]),
                **pso_options
            )
            ee = calculate_end_effector_position(link_vectors, axes, thetas)[-1]
            if np.linalg.norm(np.array(target_path[0]) - np.array(ee)) <= max_error:
                joint_path.append(thetas.tolist())
                initial_thetas_found = True
                break
        if not initial_thetas_found:
            continue

        success = True
        for i in range(1, len(target_path)):
            prev_joint = joint_path[-1]
            target = np.array(target_path[i])

            res = minimize(
                objective_function, prev_joint, # Use previous joint as initial guess
                args=(link_vectors, axes, target),
                bounds=bounds,
                method='SLSQP', # Or 'trust-constr'
                # jac=calculate_jacobian, # Uncomment if you have a proper analytical Jacobian
                options=local_opt_options
            )
            ee = calculate_end_effector_position(link_vectors, axes, res.x)[-1]
            if np.linalg.norm(target - np.array(ee)) > max_error:
                # Local optimization failed for this step. Try PSO or break.
                # Here's where you could insert a more sophisticated retry with PSO
                # as part of the main loop, not just in the "rescue" segment.
                # For simplicity here, we'll just fail the path.
                success = False
                break
            joint_path.append(res.x.tolist())

        if success:
            return joint_path

        # Your existing best_run_path logic for partial success
        if len(joint_path) > len(overall_best_path): # Changed from best_run_length
            overall_best_path = joint_path.copy()

    # --- Try hybrid rescue (if main loop failed to complete the path) ---
    if overall_best_path and len(overall_best_path) < len(target_path):
        remaining_targets = target_path[len(overall_best_path):]
        # Note: solve_segment_hybrid starts with PSO for its first point
        seg = solve_segment_hybrid(remaining_targets, overall_best_path[-1])
        if seg:
            overall_best_path.extend(seg[1:]) # seg[0] is initial_guess, already in path
            return overall_best_path
        # If rescue also fails, the best partial path found so far will be returned.
        return overall_best_path # Return the longest successful partial path

    raise RuntimeError("Failed to solve any portion of the path.")


# You already have this function, just ensure it's present and correct
def compute_jacobian(link_vectors: List[List[float]], axes: List[List[float]], thetas: List[float]) -> np.ndarray:
    num_joints = len(link_vectors)
    J = np.zeros((3, num_joints))
    positions = calculate_end_effector_position(link_vectors, axes, thetas)
    end_effector_pos = positions[-1]
    rotation = R.identity()

    for i in range(num_joints):
        # Rotate the axis of rotation by the accumulated rotation of previous joints
        current_axis_rotated = rotation.apply(axes[i])
        p_i = positions[i] # Position of the current joint
        J[:, i] = np.cross(current_axis_rotated, end_effector_pos - p_i)

        # Update the accumulated rotation for the next joint
        if not np.allclose(axes[i], [0, 0, 0]): # Avoid creating rotation from zero axis
            rotation *= R.from_rotvec(np.array(axes[i]) * thetas[i])
    return J

# You already have this function, ensure it's present and correct
def standard_jacobian_path(
    target_path: List[List[float]],
    link_vectors: List[List[float]],
    axes: List[List[float]],
    max_iterations: int = 100,
    damping: float = 0.01,
    tolerance: float = 1e-6,
    initial_angles: Optional[List[float]] = None
) -> List[List[float]]:
    num_joints = len(link_vectors)
    joint_angles_path = []
    # Start with initial_angles if provided, otherwise zeros
    current_angles = np.array(initial_angles if initial_angles is not None else [0.0] * num_joints)

    # Define angle limits for clipping, if available (assuming a global variable or passing it)
    # If angle_limits is not a global variable, you'll need to pass it to this function.
    # For now, using a placeholder for angle limits if they are not passed/global.
    # For a robust solution, consider adding angle_limits to the function signature.
    temp_angle_limits = [(-2 * np.pi, 2 * np.pi)] * num_joints # Default wide limits
    if 'angle_limits' in globals(): # Check if global angle_limits exists
        temp_angle_limits = angle_limits
    lb = np.array([limit[0] for limit in temp_angle_limits])
    ub = np.array([limit[1] for limit in temp_angle_limits])


    for target in target_path:
        for _ in range(max_iterations):
            positions = calculate_end_effector_position(link_vectors, axes, current_angles.tolist())
            current_pos = positions[-1]
            error = np.array(target) - current_pos
            if np.linalg.norm(error) < tolerance:
                break

            J = compute_jacobian(link_vectors, axes, current_angles.tolist())

            # Damped Least Squares
            JT = J.T
            JJt = J @ JT
            # Add damping to the diagonal of JJt
            dq = JT @ np.linalg.inv(JJt + (damping ** 2) * np.eye(3)) @ error

            current_angles += dq

            # Clip joint angles to limits
            current_angles = np.clip(current_angles, lb, ub)

        joint_angles_path.append(current_angles.copy().tolist())

    return joint_angles_path

def pure_pso_path(
    target_path: List[List[float]],
    link_vectors: List[List[float]],
    axes: List[List[float]],
    angle_limits: Optional[List[Tuple[float, float]]] = None,
    swarmsize: int = 300,
    maxiter: int = 100,
    min_error_tolerance: float = 0.01
) -> List[List[float]]:

    joint_angles_path = []
    num_joints = len(link_vectors)

    if angle_limits is None:
        angle_limits = [(-2 * np.pi, 2 * np.pi)] * num_joints

    lb = [limit[0] for limit in angle_limits]
    ub = [limit[1] for limit in angle_limits]

    pso_options = {'swarmsize': swarmsize, 'maxiter': maxiter}

    for i, target_point in enumerate(target_path):
        # Initial guess for PSO can be random or zeros, as it doesn't use path continuity
        # Using a random guess for each point emphasizes the 'pure' aspect.
        # This will be computationally expensive.
        # initial_guess = np.random.uniform(lb, ub, num_joints) # More 'pure' PSO for each point

        # For slight pragmatism in path following, you could pass the previous solution
        # to PSO *as an initial particle*, but PSO still explores globally from there.
        # However, for a true 'pure' comparison, independent searches are better.
        # Let's stick to truly independent searches for stark comparison.
        thetas, _ = pso(
            objective_function, lb, ub,
            args=(link_vectors, axes, target_point),
            **pso_options,
            # Add specific tolerances for pso.pso to converge
            minfunc=min_error_tolerance,
            minstep=1e-4 # Minimum step size for particles
        )

        ee_pos = calculate_end_effector_position(link_vectors, axes, thetas.tolist())[-1]
        error = np.linalg.norm(np.array(target_point) - np.array(ee_pos))

        if error > min_error_tolerance:
            print(f"Warning: Pure PSO failed to reach target {i} with desired accuracy. Error: {error:.4f}")
            # Decide whether to raise error or append best effort.
            # For comparison, we'll append best effort but note it.
            # raise RuntimeError(f"Pure PSO failed to reach target {i} with desired accuracy. Error: {error:.4f}")
        joint_angles_path.append(thetas.tolist())

    return joint_angles_path

def calculate_joint_velocity_smoothness(joint_angles_path: List[List[float]]) -> float:
    """
    Calculates the average L2 norm of joint velocity (change in angles)
    between consecutive steps to quantify path smoothness. Lower is smoother.
    """
    if len(joint_angles_path) < 2:
        return 0.0 # Cannot calculate smoothness for a single point or empty path

    joint_angles_np = np.array(joint_angles_path)
    # Calculate differences between consecutive angle sets
    delta_angles = np.diff(joint_angles_np, axis=0)
    # Calculate L2 norm for each step's joint velocity vector
    l2_norms = np.linalg.norm(delta_angles, axis=1)
    # Return the average L2 norm
    return np.mean(l2_norms)

def plot_comparison_metrics(metrics_data: dict, title: str = "Robot IK Algorithm Comparison"):
    """
    Plots a bar chart comparing performance metrics of different IK algorithms.
    metrics_data format: {'Algorithm Name': {'Metric1': value, 'Metric2': value, ...}}
    """
    metrics = list(list(metrics_data.values())[0].keys()) # Get metric names from first algorithm
    algorithms = list(metrics_data.keys())

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1: # If only one metric, axes is not an array
        axes = [axes]

    colors = plt.cm.tab10.colors # Use a colormap for distinct bars

    for i, metric_name in enumerate(metrics):
        values = [metrics_data[algo].get(metric_name, 0) for algo in algorithms]
        ax = axes[i]
        bars = ax.bar(algorithms, values, color=colors[:len(algorithms)])
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        # --- START OF FIX ---
        # Rotate x labels directly, and set their alignment
        for tick_label in ax.get_xticklabels():
            tick_label.set_rotation(45)
            tick_label.set_horizontalalignment('right')
        # --- END OF FIX ---
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.show()

def plot_all_comparisons_in_one_figure(
    algorithm_results: dict,
    target_path: List[List[float]],
    link_vectors: List[List[float]],
    axes: List[List[float]],
    num_dof: int,
    max_error_tolerance: float = 0.01
):

    num_algorithms = len(algorithm_results)
    if num_algorithms == 0:
        print("No algorithm results to plot.")
        return

    # Determine layout: (number of algorithms) rows, 2 columns (Joint Angles, End-effector Error)
    fig, axes_grid = plt.subplots(num_algorithms, 2, figsize=(16, 5 * num_algorithms),
                                  squeeze=False) # squeeze=False ensures axes_grid is always 2D

    colors_joints = plt.cm.get_cmap('tab10', num_dof) # Colormap for joint angles
    colors_lines = ['red', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink', 'gray'] # For algorithm paths

    row_idx = 0
    for algo_name, data in algorithm_results.items():
        joint_angles = data.get('joint_angles')
        errors = data.get('errors')

        # Skip if no data for this algorithm (e.g., if it failed completely)
        if joint_angles is None or errors is None:
            print(f"Skipping plot for {algo_name} due to missing data.")
            continue

        joint_angles_np = np.array(joint_angles)
        num_points, num_joints = joint_angles_np.shape

        # --- Plot Joint Angles (Left Column) ---
        ax_joint = axes_grid[row_idx, 0]
        for j in range(num_joints):
            ax_joint.plot(joint_angles_np[:, j], label=f'Joint {j+1}',
                          color=colors_joints(j), linewidth=1.5)
        ax_joint.set_title(f'{algo_name} - Joint Angles')
        ax_joint.set_xlabel("Time Step")
        ax_joint.set_ylabel("Joint Angle (rad)")
        ax_joint.legend(loc='upper right', ncol=2, fontsize='small')
        ax_joint.grid(True, linestyle='--', alpha=0.6)

        # --- Plot End-effector Error (Right Column) ---
        ax_error = axes_grid[row_idx, 1]
        ax_error.plot(errors, color='red', linewidth=2.5, label='End-effector Error')
        ax_error.axhline(y=max_error_tolerance, color='gray', linestyle='--', label='Tolerance Limit')
        ax_error.set_title(f'{algo_name} - End-effector Error')
        ax_error.set_xlabel("Time Step")
        ax_error.set_ylabel("Positional Error (m)")
        ax_error.legend(loc='upper right', fontsize='small')
        ax_error.grid(True, linestyle='--', alpha=0.6)
        # Set y-limit for error plot for better comparison, max(max_error, highest error in all plots)
        ax_error_ylim = max(max(errors), max_error_tolerance * 1.5) # ensure tolerance line is visible
        ax_error.set_ylim(0, ax_error_ylim)

        row_idx += 1

    fig.suptitle(f'Comparison of Inverse Kinematics Algorithms for {num_dof}-DOF Robot', fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()

    # Optional: Plot 3D paths in a single figure
    fig_paths = go.Figure()

    # Add target path
    fig_paths.add_trace(go.Scatter3d(
        x=np.array(target_path)[:, 0], y=np.array(target_path)[:, 1], z=np.array(target_path)[:, 2],
        mode='lines', line=dict(color='blue', width=7), name='Target Path',
        hovertemplate='Target X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))

    current_line_color_idx = 0
    for algo_name, data in algorithm_results.items():
        joint_angles = data.get('joint_angles')
        if joint_angles and len(joint_angles) > 0:
            ee_path = [calculate_end_effector_position(link_vectors, axes, angles)[-1] for angles in joint_angles]
            fig_paths.add_trace(go.Scatter3d(
                x=np.array(ee_path)[:, 0], y=np.array(ee_path)[:, 1], z=np.array(ee_path)[:, 2],
                mode='lines', line=dict(color=colors_lines[current_line_color_idx % len(colors_lines)], width=3), name=f'{algo_name} Achieved Path',
                hovertemplate=f'{algo_name} X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
            ))
            current_line_color_idx += 1


    fig_paths.update_layout(
        title=f'Achieved End-Effector Paths vs. Target Path for {num_dof}-DOF Arm',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube' # Keep aspect ratio for spatial accuracy
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig_paths.show()

    # --- Main Execution for Comparison ---







num_dof = 7 # Change to 6, 7, 8 etc.
link_length_base = 0.5
link_length_variation = 0.2

link_vectors = []
axes = []
for i in range(num_dof):
    link_vectors.append([0, 0, link_length_base + (link_length_variation * (i % 2))])
    if i % 2 == 0:
        axes.append([0, 0, 1]) # Z-axis rotation
    else:
        axes.append([0, 1, 0]) # Y-axis rotation

# Normalize axes (essential)
axes = [np.array(axis) / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else np.array(axis) for axis in axes]

# Define angle limits for all joints
angle_limits = [(-np.pi, np.pi)] * num_dof # Example: full 360-degree range for all joints

print(f"Robot configured with {num_dof} Degrees of Freedom.")
print(f"Link Vectors: {link_vectors}")
print(f"Axes: {axes}")

# 2. Generate Target Path
# path = generate_random_loop_path(points=100, seed=42) # Use a fixed seed for reproducible paths
path = generate_non_linear_path(points=100) # Using your non-linear path from the provided code

# Optionally visualize the path and workspace (useful for debugging/understanding)
# plot_workspace_with_path(link_vectors, axes, path, title=f"Desired Path for {num_dof}-DOF Robot")


# 3. Algorithms Comparison
comparison_results_detailed = {} # Store joint_angles and errors for plotting
max_error_tolerance = 0.01 # Define a consistent tolerance for all methods


# --- Run Your Hybrid Method (fast_hybrid_path2_optimized) ---
print("\n--- Running Your Hybrid Method (fast_hybrid_path2_optimized) ---")
hybrid_joint_angles = None
hybrid_errors = None
try:
    start_time_hybrid = time.time()
    hybrid_joint_angles = fast_hybrid_path2(path, link_vectors, axes, angle_limits)
    end_time_hybrid = time.time()
    hybrid_runtime = end_time_hybrid - start_time_hybrid
    hybrid_errors = accuracy_graph(hybrid_joint_angles, path, link_vectors, axes)
    hybrid_avg_error = np.mean(hybrid_errors)
    hybrid_max_error = np.max(hybrid_errors)
    hybrid_smoothness = calculate_joint_velocity_smoothness(hybrid_joint_angles)
    hybrid_success_percentage = (len(hybrid_joint_angles) / len(path)) * 100

    print(f"Hybrid Runtime: {hybrid_runtime:.4f} seconds")
    print(f"Hybrid Avg Error: {hybrid_avg_error:.6f}")
    print(f"Hybrid Max Error: {hybrid_max_error:.6f}")
    print(f"Hybrid Smoothness (Avg Joint Velocity): {hybrid_smoothness:.6f}")
    print(f"Hybrid Success Rate: {hybrid_success_percentage:.2f}%")

    comparison_results_detailed['Your Hybrid'] = {
        'joint_angles': hybrid_joint_angles,
        'errors': hybrid_errors,
        'Runtime (s)': hybrid_runtime,
        'Avg Error': hybrid_avg_error,
        'Max Error': hybrid_max_error,
        'Smoothness': hybrid_smoothness,
        'Success Rate (%)': hybrid_success_percentage
    }
    # animate_robot(link_vectors, axes, hybrid_joint_angles) # Uncomment to animate
except RuntimeError as e:
    print(f"Your Hybrid Method failed: {e}")
    comparison_results_detailed['Your Hybrid'] = {
        'joint_angles': None, 'errors': None,
        'Runtime (s)': np.nan, 'Avg Error': np.nan, 'Max Error': np.nan,
        'Smoothness': np.nan, 'Success Rate (%)': 0
    }


# --- Run DLS Jacobian Method ---
print("\n--- Running DLS Jacobian Method ---")
dls_joint_angles = None
dls_errors = None
try:
    start_time_dls = time.time()
    initial_angles_dls = np.zeros(num_dof).tolist() # Start DLS from zero angles
    dls_joint_angles = standard_jacobian_path(
        path, link_vectors, axes,
        max_iterations=200,
        damping=0.05,
        tolerance=max_error_tolerance,
        initial_angles=initial_angles_dls
    )
    end_time_dls = time.time()
    dls_runtime = end_time_dls - start_time_dls
    dls_errors = accuracy_graph(dls_joint_angles, path, link_vectors, axes)
    dls_avg_error = np.mean(dls_errors)
    dls_max_error = np.max(dls_errors)
    dls_smoothness = calculate_joint_velocity_smoothness(dls_joint_angles)
    dls_success_percentage = (np.sum(np.array(dls_errors) <= max_error_tolerance) / len(path)) * 100

    print(f"DLS Runtime: {dls_runtime:.4f} seconds")
    print(f"DLS Avg Error: {dls_avg_error:.6f}")
    print(f"DLS Max Error: {dls_max_error:.6f}")
    print(f"DLS Smoothness (Avg Joint Velocity): {dls_smoothness:.6f}")
    print(f"DLS Success Rate: {dls_success_percentage:.2f}%")

    comparison_results_detailed['DLS Jacobian'] = {
        'joint_angles': dls_joint_angles,
        'errors': dls_errors,
        'Runtime (s)': dls_runtime,
        'Avg Error': dls_avg_error,
        'Max Error': dls_max_error,
        'Smoothness': dls_smoothness,
        'Success Rate (%)': dls_success_percentage
    }
    # animate_robot(link_vectors, axes, dls_joint_angles, obstacles) # Uncomment to animate

except Exception as e:
    print(f"DLS Jacobian Method failed: {e}")
    comparison_results_detailed['DLS Jacobian'] = {
        'joint_angles': None, 'errors': None,
        'Runtime (s)': np.nan, 'Avg Error': np.nan, 'Max Error': np.nan,
        'Smoothness': np.nan, 'Success Rate (%)': 0
    }


# --- Run Pure PSO Method ---
print("\n--- Running Pure PSO Method ---")
pso_joint_angles = None
pso_errors = None
try:
    start_time_pso = time.time()
    pso_joint_angles = pure_pso_path(path, link_vectors, axes, angle_limits,
                                      swarmsize=500,
                                      maxiter=200,
                                      min_error_tolerance=max_error_tolerance)
    end_time_pso = time.time()
    pso_runtime = end_time_pso - start_time_pso
    pso_errors = accuracy_graph(pso_joint_angles, path, link_vectors, axes)
    pso_avg_error = np.mean(pso_errors)
    pso_max_error = np.max(pso_errors)
    pso_smoothness = calculate_joint_velocity_smoothness(pso_joint_angles)
    pso_success_percentage = (np.sum(np.array(pso_errors) <= max_error_tolerance) / len(path)) * 100


    print(f"Pure PSO Runtime: {pso_runtime:.4f} seconds")
    print(f"Pure PSO Avg Error: {pso_avg_error:.6f}")
    print(f"Pure PSO Max Error: {pso_max_error:.6f}")
    print(f"Pure PSO Smoothness (Avg Joint Velocity): {pso_smoothness:.6f}")
    print(f"Pure PSO Success Rate: {pso_success_percentage:.2f}%")

    comparison_results_detailed['Pure PSO'] = {
        'joint_angles': pso_joint_angles,
        'errors': pso_errors,
        'Runtime (s)': pso_runtime,
        'Avg Error': pso_avg_error,
        'Max Error': pso_max_error,
        'Smoothness': pso_smoothness,
        'Success Rate (%)': pso_success_percentage
    }
    # animate_robot(link_vectors, axes, pso_joint_angles) # Uncomment to animate

except Exception as e:
    print(f"Pure PSO Method failed: {e}")
    comparison_results_detailed['Pure PSO'] = {
        'joint_angles': None, 'errors': None,
        'Runtime (s)': np.nan, 'Avg Error': np.nan, 'Max Error': np.nan,
        'Smoothness': np.nan, 'Success Rate (%)': 0
    }


# 4. Generate Combined Plots for Research Paper
print("\n--- Generating Combined Comparison Plots ---")
# Extract only the summary metrics for the bar chart
summary_metrics_for_bar_chart = {
    algo_name: {k: v for k, v in data.items() if k not in ['joint_angles', 'errors']}
    for algo_name, data in comparison_results_detailed.items()
}

plot_comparison_metrics(summary_metrics_for_bar_chart, title=f"Summary of IK Algorithm Performance for {num_dof}-DOF Arm")

# Call the new function to plot all joint angles and errors in one figure
plot_all_comparisons_in_one_figure(
    comparison_results_detailed,
    path,
    link_vectors,
    axes,
    num_dof,
    max_error_tolerance
)