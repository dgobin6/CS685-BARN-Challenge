import torch 
import time
import rospy
import argparse
import numpy as np

from gazebo_simulation import GazeboSimulation
from geometry_msgs.msg import Twist
from gym.spaces import Box

def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

class Policy(torch.nn.Module): 
    def __init__(self, n_input, n_output): 
        super(Policy,self).__init__()
        layers = []
        layers.append(torch.nn.Conv1d(in_channels=n_input, out_channels=256, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, padding=0))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Flatten(start_dim=0))
        layers.append(torch.nn.Linear(256, 256))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(256, n_output))
        layers.append(torch.nn.ReLU())

        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        # x [seq_len, state_dim]
        print(f'DEBUG: {x.shape}')
        x = x.unsqueeze(0)
        x = x.permute(1, 0) # [state_dim, seq_len]
        a = self.net(x)
        return a

class PolicyMotionController():
    def __init__(self, goal_pos, initial_pos, file_path, gazebo_sim):
        
        #load policy
        n_input = 724
        n_output = 2
        self.policy = Policy(n_input, n_output).to('cpu')
        self.policy.load_state_dict(torch.load(file_path))
        self.gazebosim = gazebo_sim

        #parameters
        self.laser_clip = 5
        self.min_v = -1
        self.max_v = 2
        self.min_w = -3.14
        self.max_w = 3.14
        self.time_step = .2

        #values
        self.world_frame_goal = (initial_pos[0] + goal_pos[0], initial_pos[1]+goal_pos[1])
        self.action_space = Box(
            low=np.array([self.min_v, self.min_w]),
            high=np.array([self.max_v, self.max_w]),
            dtype=np.float32
        )

        #ROS
        self.cmd_vel = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=1)
        rospy.init_node('controller', anonymous=True, log_level=rospy.FATAL)

    def observe(self, laser_scan, pos, psi, action):
        #laser scan scaling
        laser_scan = np.array(laser_scan.ranges)
        laser_scan[laser_scan > self.laser_clip] = self.laser_clip
        laser_scan = (laser_scan - self.laser_clip/2.) / self.laser_clip * 2 

        #position framing and scaling
        R_r2i = np.matrix([[np.cos(psi), -np.sin(psi), pos.x], [np.sin(psi), np.cos(psi), pos.y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.matrix([[self.world_frame_goal[0]], [self.world_frame_goal[1]], [1]])
        pr = np.matmul(R_i2r, pi)
        lg = np.array([pr[0,0], pr[1, 0]])
        goal_pos = lg/5.0 - 1
        
        #action scaling
        bias = (self.action_space.high + self.action_space.low)/2.
        scale = (self.action_space.high - self.action_space.low)/2.
        action = (action-bias)/scale
        
        obs = [laser_scan, goal_pos, action]

        obs = np.concatenate(obs)

        return obs
    
    def act(self, observation, ctime):
        observation = torch.from_numpy(observation).float().to('cpu')
        action = (self.policy(observation).data.detach().cpu().numpy())

        linear, angular = action
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear
        cmd_vel_value.angular.z = angular

        self.gazebosim.unpause()
        self.cmd_vel.publish(cmd_vel_value)
        self.wait_on_timestep(ctime)
        self.gazebosim.pause()

        return action

    def wait_on_timestep(self, stime):
        current_time = rospy.get_time()
        while current_time - stime < self.time_step:
            time.sleep(0.01)
            current_time = rospy.get_time()

    def move(self):
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        
        start_time = curr_time
        collided = False
        
        action = np.asarray([0,0])

        while compute_distance(self.world_frame_goal, curr_coor) > 1 and not collided and curr_time - start_time < 100:
            #time
            curr_time = rospy.get_time()
            #get laser scan
            self.gazebosim.unpause()
            ls = self.gazebosim.get_laser_scan()
            
            #get pose
            pose =self.gazebosim.get_model_state().pose
            pos = pose.position
            q1 = pose.orientation.x
            q2 = pose.orientation.y
            q3 = pose.orientation.z
            q0 = pose.orientation.w
            psi = np.arctan2(2* (q0*q3 + q1*q2), (1-2*(q2**2+q3**2)))
            assert -np.pi <= psi <= np.pi, psi

            self.gazebosim.pause()
            observation = self.observe(ls, pos, psi, action)
            action = self.act(observation, curr_time)



if __name__== '__main__':
    parser = argparse.ArgumentParser(description = 'move jackal via policy')
    parser.add_argument('--ipos', dest='initial_pos')
    parser.add_argument('--gpos', dest='goal_pos')
    parser.add_argument('--policy_file', dest='policy_file')
    args = parser.parse_args()


    init_pos = np.asarray(args.initial_pos.split(" "), dtype=float)
    goal_pos = np.asarray(args.goal_pos.split(" "), dtype=float)

    gazebo_sim = GazeboSimulation(init_position=init_pos)
    pmc = PolicyMotionController(goal_pos, init_pos, args.policy_file, gazebo_sim)

    pmc.move()

