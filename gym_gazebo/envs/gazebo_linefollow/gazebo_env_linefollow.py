import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected


    def erode_image(self, imgrey):
        
        '''
            @brief Erodes the image to remove noise

            @param imgrey : Image data from ROS

        '''

        kernel_size = 3  # Adjust the size as needed
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply erosion on the binary image using the binary mask
        eroded_image = cv2.erode(imgrey, kernel)
        return eroded_image



    def center_of_road(self, image, height, radius):

        '''
            @brief Finds the center of the road

            @param image : Image data from ROS, should be greyscaled
        '''


        #Twentieth Row from bottom
        row_of_pixels = image[height - radius, :]

        #Returns a tuple, one for row and one column. Values of each element hold
        #the index of a pixel on the original array that passed the condition
        #We only passed it one row, so we only care about columns, [0] is columns
        non_zero_indices = np.where(row_of_pixels > 0)[0]

        #If no non-zero values found
        if len(non_zero_indices) == 0:
            return 0

        #Get median value from these list of indices, round incase it is non-int
        median = np.median(non_zero_indices)

        return int(median)


    def find_road(self, cv_image):

        '''
            @brief Finds the road in the cv2 image
            
            @param cv_img : Image data from ROS
        '''

        # position of road in image
        segment_idx = None
        # Dimensions of image
        height, width, _ = cv_image.shape
        # Radius of circle to be drawn on center of road
        radius = 25

        # Process image to recognize road
        imgrey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _ , masked_img = cv2.threshold(imgrey,127,255,cv2.THRESH_BINARY_INV)
        eroded_img = self.erode_image(masked_img)

        #Slice image into segments
        segment_width = width // 10
        for i in range(1, 10):
            
            # Horizontal position of line dividing two segments
            x_pixel = segment_width * i

            # Draw divider line
            cv2.line(cv_image, (x_pixel, 0), (x_pixel, height), (0, 0, 0), 2)

        # Find center of road
        centerx_of_circle = self.center_of_road(eroded_img, height, radius)

        # Draw circle on road
        cv2.circle(cv_image,
                    (centerx_of_circle, height - radius),
                    radius,
                    (0,0,255),
                    thickness=cv2.FILLED)

        # Determine what segment the center of the road is in

        # If no road found
        if centerx_of_circle == 0:
            return segment_idx, cv_image
        else:
            segment_idx = math.floor(centerx_of_circle / segment_width)
            return segment_idx, cv_image

    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        NUM_BINS = 3
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        # Find road
        road_idx, image = self.find_road(cv_image)

        # Update state array
        # Determine if the road has not been visible for 30 frames
        if road_idx != None:
            state[road_idx] = 1
            self.timeout = 0
        else:
            self.timeout += 1

        if self.timeout >= 30:
            done = True

        # Display state vector
        ((text_width, text_height), _) = cv2.getTextSize(str(state), cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        cv2.putText(image, str(state), (10,text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)

        # Display image        
        cv2.imshow("Image window", image)
        cv2.waitKey(1)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
