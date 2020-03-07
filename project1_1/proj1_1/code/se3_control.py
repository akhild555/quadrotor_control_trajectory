import numpy as np
from scipy.spatial.transform import Rotation
from flightsim.crazyflie_params import quad_params

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE

        # gain matrices
        self.posfactor = 1 # position gains scaling factor
        self.attfactor = 1 # attitude gains scaling factor
        self.Kd = np.diag(np.array([3.2*self.posfactor, 3.2*self.posfactor, 19.5]))
        self.Kp = np.diag(np.array([4*self.posfactor, 4*self.posfactor, 38]))
        self.Kr = np.diag(np.array([65*self.attfactor, 65*self.attfactor, 76]))
        self.Kw = np.diag(np.array([9*self.attfactor, 9*self.attfactor, 9]))


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # Implement Geometric Non-Linear Controller

        # Calculate F_des
        weight = np.array([[0],[0],[self.mass*self.g]])
        r_des = flat_output['x_ddot'].reshape(-1,1) - self.Kd @ (state['v'].reshape(-1,1)-flat_output['x_dot'].reshape(-1,1)) - self.Kp @ (state['x'].reshape(-1,1)-flat_output['x'].reshape(-1,1))
        F_des = self.mass * r_des + weight

        # Calculate u1
        rotM = Rotation.from_quat(state['q']).as_matrix()
        b3 = rotM @ np.array([[0],[0],[1]])
        u1 = b3.T @ F_des


        # Calculate R_des
        b3_des = np.divide(F_des,np.linalg.norm(F_des))
        a_phi = np.array([[np.cos(flat_output['yaw'])],[np.sin(flat_output['yaw'])],[0]])
        crs = np.cross(b3_des.T,a_phi.T)
        b2_des = np.divide(np.cross(b3_des.T,a_phi.T),np.linalg.norm(np.cross(b3_des.T,a_phi.T)))
        b2_des = b2_des.reshape(-1,1)
        col1 = np.cross(b2_des.T,b3_des.T)
        R_des = np.concatenate((col1.T,b2_des,b3_des), axis=1)
        # R_des = np.eye(3)
        # error

        vee = R_des.T @ rotM - rotM.T @ R_des
        vec = np.array([vee[2,1], vee[0,2], vee[1,0]])
        er = 0.5*vec.reshape(-1,1)

        # Calculate u2
        u2 = self.inertia @ (-self.Kr @ er - self.Kw @ state['w'].reshape(-1,1))

        # Calculate Forces
        k = self.k_drag / self.k_thrust
        L = self.arm_length
        M = np.array([[1, 1, 1, 1],
                      [0, L, 0, -L],
                      [-L, 0, L, 0],
                      [k, -k, k, -k]])
        uVec = np.concatenate((u1,u2), axis=0)
        Forces = np.linalg.inv(M) @ uVec
        Forces = Forces.clip(min=0)

        angV = np.sqrt(np.divide(Forces,self.k_thrust)) # convert to motor speeds



        # Outputs
        cmd_motor_speeds[0] = angV[0]
        cmd_motor_speeds[1] = angV[1]
        cmd_motor_speeds[2] = angV[2]
        cmd_motor_speeds[3] = angV[3]
        cmd_thrust = u1[0][0]
        # cmd_thrust = cmd_thrust.clip(min=0)
        cmd_moment = u2
        cmd_q = Rotation.from_dcm(R_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input