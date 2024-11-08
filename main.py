import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image, LaserScan
import cv2
import cv_bridge
import numpy as np
import time

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        self.state_sub = self.create_subscription(State, '/uav1/mavros/state', self.state_cb, 10)
        self.local_pos_pub = self.create_publisher(PoseStamped, '/uav1/mavros/setpoint_position/local', 10)
        self.image_sub = self.create_subscription(Image, '/uav1/camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/uav1/laser/scan', self.laser_callback, 10)
        self.arming_client = self.create_client(CommandBool, '/uav1/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/uav1/mavros/set_mode')
        
        # Initialize variables
        self.current_state = State()
        self.pose = PoseStamped()
        self.cv_bridge = cv_bridge.CvBridge()
        self.timer = self.create_timer(0.1, self.fly_and_search_objects)
        self.is_flying = False
        self.altitude = 2.0
        self.search_mode = True
        self.laser_data = []
        self.safe_distance = 2.0
        self.detected_cubes = set()
        self.cube_count = 8

    def state_cb(self, msg):
        self.current_state = msg

    def laser_callback(self, msg):
        self.laser_data = msg.ranges

    def fly_and_search_objects(self):
        if not self.current_state.connected:
            self.get_logger().info("Waiting for connection to the drone...")
            return
        
        if self.current_state.mode != "OFFBOARD":
            self.set_offboard_mode()
        
        if not self.current_state.armed:
            self.arm_drone()
        
        if not self.is_flying:
            # Set initial position
            self.pose.pose.position.z = self.altitude
            # Ensure x and y are initialized correctly
            if not hasattr(self.pose.pose.position, 'x'):
                self.pose.pose.position.x = 0.0
                self.pose.pose.position.y = 0.0
            
            # Mark as flying
            self.is_flying = True
            self.get_logger().info("Drone is taking off to search for cubes...")

        if len(self.detected_cubes) < self.cube_count:
            if not self.is_obstacle_ahead():
                # Move forward to search for cubes
                if not hasattr(self, 'hover_start_time'):
                    # Start moving in the room
                    if abs(self.pose.pose.position.x) < 15 and abs(self.pose.pose.position.y) < 15:
                        # Move forward in x direction
                        if abs(self.pose.pose.position.x) < 15:
                            self.pose.pose.position.x += 0.2
                        else: 
                            # Turn left when reaching x boundary
                            self.pose.pose.position.y += 0.2
                    else: 
                        # Turn left when reaching y boundary
                        if abs(self.pose.pose.position.y) >= 15:
                            self.pose.pose.position.x -= 0.2
                
                else:
                    # Check hovering time over detected cube
                    elapsed_time = time.time() - getattr(self, 'hover_start_time', time.time())
                    if elapsed_time >= 10:
                        delattr(self, 'hover_start_time')  # Reset hover timer after hovering

            else:
                # If an obstacle is detected, avoid it.
                self.avoid_obstacle()

            # Publish drone position in ROS.
            distance_to_cube = None
            
            for cube in list(self.detected_cubes):
                color, cx, cy = cube.split('_')
                distance_to_cube = float(cy) / 100.0
            
                if distance_to_cube is not None:
                    cv2.putText(frame, f"Distance to cube: {distance_to_cube:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Send drone position to ROS.
            self.local_pos_pub.publish(self.pose)
            self.explore_environment()


        else:
            # If all cubes are found - land.
            if hasattr(self, 'hover_start_time'):
                delattr(self, 'hover_start_time')
            else: 
                # Land at starting position.
                if abs(self.pose.pose.position.x) > 0 or abs(self.pose.pose.position.y) > 0 or abs(self.pose.pose.position.z) > 0:
                    if abs(self.pose.pose.position.z) > 0:
                        # Decrease height to zero gradually.
                        print(f"Current altitude: {self.pose.pose.position.z:.3f}")
                        self.pose.pose.position.z -= 0.1  
                    else: 
                        # Land the drone.
                        print("Landing...")
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id="base_link"
                        pose_msg.point.z -= 0.5  
                        print("Landed at the start position.")
                        rclpy.shutdown()

    def is_obstacle_ahead(self):
        filtered_laser_data = [d for d in self.laser_data if d > 0]
        return any(distance < (self.safe_distance + 1) for distance in filtered_laser_data)

    def avoid_obstacle(self):
        filtered_laser_data = [d for d in self.laser_data if d > 0]
        
        if filtered_laser_data:  
            min_distance = min(filtered_laser_data)
            min_index = filtered_laser_data.index(min_distance)

            angle_change = -0.2 if min_index < len(filtered_laser_data) // 2 else +0.2  
            
            new_yaw_angle = (self.pose.pose.orientation.z + angle_change) % (2 * np.pi)
            # Update orientation.
            self.pose.pose.orientation.z = new_yaw_angle

    def set_offboard_mode(self):
        if self.set_mode_client.wait_for_service(timeout_sec=1.0):
            set_mode_req = SetMode.Request()
            set_mode_req.custom_mode = 'OFFBOARD'
            future_response = self.set_mode_client.call_async(set_mode_req)
            
            future_response.add_done_callback(lambda response: print("OFFBOARD mode set" if response.result().mode_sent else "Failed to set OFFBOARD mode"))

    def arm_drone(self):
        if self.arming_client.wait_for_service(timeout_sec=1.0):
            arm_req = CommandBool.Request()
            arm_req.value = True
            
            future_response =self.arming_client.call_async(arm_req)
            
            future_response.add_done_callback(lambda response: print("Drone armed" if response.result().success else "Failed to arm drone"))

    def image_callback(self, msg):
        global frame
        
        frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color in ['red', 'green', 'blue', 'yellow']:
            mask = self.detect_color(hsv_frame, color)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2
                    
                    distance_to_cube = max(0.1,(self.calculate_distance_to_object(w,h)))
                    cube_id=f"{color}_{cx}_{cy}"
                    
                    if cube_id not in (self.detected_cubes):
                        print(f"Cube detected: {color}, distance: {distance_to_cube:.2f}m")
                        cv2.putText(frame,f"{color} cube detected", (cx-20 , cy-20),cv2.FONT_HERSHEY_SIMPLEX ,0.5 ,(255 ,255 ,255),1)
                        
                        hover_start_time=time.time()  
                        setattr(self,'hover_start_time',hover_start_time)  
                        
                        # Add cube to detected cubes set.
                        self.detected_cubes.add(cube_id)
                        
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0 ,255 ,0),2)
                    cv2.putText(frame,f"{color} cube,{distance_to_cube:.2f}m",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX ,0.5,(255 ,0 ,0),1)

        cv2.imshow("Frame", frame)

    def detect_color(self,hsv_frame,color):
        lower_upper_color_map={
            'red':(np.array([0 ,120 ,70]),np.array([10 ,255 ,255])),
            'green':(np.array([36 ,100 ,100]),np.array([86 ,255 ,255])),
            'blue':(np.array([94 ,80 ,2]),np.array([126 ,255 ,255])),
            'yellow':(np.array([15 ,100 ,100]),np.array([35 ,255 ,255]))
        }
        
        lower_upper=lower_upper_color_map[color]
        
        return cv2.inRange(hsv_frame,*lower_upper)

    def calculate_distance_to_object(self,width,height):
        focal_length=500  
        real_width=0.1  
        
        return (focal_length*real_width)/width

    def land_drone(self):
        print("Landing...")
        
        while abs(self.pose.pose.position.z) > 0:
            time.sleep(1)
        
        print("Landed at the start position.")

    def explore_environment(self):
        # Move in a simple search pattern (e.g., grid) with reduced speed
        if abs(self.pose.pose.position.x) < 15 and abs(self.pose.pose.position.y) < 15:
            self.pose.pose.position.x += 0.05 if abs(self.pose.pose.position.x) < 15 else 0.0
            self.pose.pose.position.y += 0.05 if abs(self.pose.pose.position.y) >= 15 else 0.0
        else:
            self.pose.pose.position.x, self.pose.pose.position.y = -15, -15

        
def main(args=None):
    rclpy.init(args=args)
        
    drone_controller=DroneController()
        
    rclpy.spin(drone_controller)
        
    drone_controller.destroy_node()
        
    rclpy.shutdown()

if __name__ == '__main__':
    main()
