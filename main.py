import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import LaserScan

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        
        self.state_sub = self.create_subscription(State, '/uav1/mavros/state', self.state_cb, 10)
        self.local_pos_pub = self.create_publisher(PoseStamped, '/uav1/mavros/setpoint_position/local', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/uav1/laser/scan', self.laser_callback, 10)

        self.arming_client = self.create_client(CommandBool, '/uav1/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/uav1/mavros/set_mode')

        self.current_state = State()
        self.pose = PoseStamped()

        self.timer = self.create_timer(0.1, self.fly_and_search_qr)

        self.is_flying = False
        self.altitude = 2.0  # Высота полета
        self.search_mode = True
        self.laser_data = []
        self.safe_distance = 2.0  # Максимально допустимое расстояние до препятствия

    def state_cb(self, msg):
        self.current_state = msg

    def laser_callback(self, msg):
        self.laser_data = msg.ranges

    def fly_and_search_qr(self):
        if not self.current_state.connected:
            self.get_logger().info("Waiting for connection to the drone...")
            return

        if self.current_state.mode != "OFFBOARD":
            self.set_offboard_mode()
        
        if not self.current_state.armed:
            self.arm_drone()

        if not self.is_flying:
            self.pose.pose.position.z = self.altitude
            self.is_flying = True
            self.get_logger().info("Drone is taking off to search for obstacles...")

        if self.search_mode:
            if self.is_obstacle_ahead():
                self.avoid_obstacle()
            else:
                self.explore_environment()
        
        self.local_pos_pub.publish(self.pose)

    def is_obstacle_ahead(self):
        return self.laser_data and any(distance < self.safe_distance for distance in self.laser_data)

    def avoid_obstacle(self):
        self.get_logger().info("Obstacle detected! Avoiding...")
        min_distance = min(self.laser_data)
        min_index = self.laser_data.index(min_distance)

        # Если препятствие слишком близко, увеличиваем высоту
        if min_distance < self.safe_distance / 2:
            self.pose.pose.position.z += 0.5  # Увеличиваем высоту
            self.get_logger().info(f"Increasing altitude to {self.pose.pose.position.z} meters.")
        else:
            # Определяем направление обхода на основе расположения минимального расстояния
            if min_index < len(self.laser_data) // 3:
                self.pose.pose.position.y += 0.5  # Уход вправо
            elif min_index > 2 * len(self.laser_data) // 3:
                self.pose.pose.position.y -= 0.5  # Уход влево
            else:
                pass  # Убираем движение назад

        self.local_pos_pub.publish(self.pose)

    def explore_environment(self):
        self.pose.pose.position.x += 0.1
        if self.pose.pose.position.x > 5:
            self.pose.pose.position.x = -5
            self.pose.pose.position.y += 0.5
            if self.pose.pose.position.y > 5:
                self.pose.pose.position.y = -5

    def set_offboard_mode(self):
        if self.set_mode_client.wait_for_service(timeout_sec=1.0):
            set_mode_req = SetMode.Request()
            set_mode_req.custom_mode = 'OFFBOARD'
            self.set_mode_client.call_async(set_mode_req)
            self.get_logger().info("OFFBOARD mode set")
        else:
            self.get_logger().warning("OFFBOARD mode service unavailable!")

    def arm_drone(self):
        if self.arming_client.wait_for_service(timeout_sec=1.0):
            arm_req = CommandBool.Request()
            arm_req.value = True
            self.arming_client.call_async(arm_req)
            self.get_logger().info("Drone armed")
        else:
            self.get_logger().warning("Arming service unavailable!")

def main(args=None):
    rclpy.init(args=args)
    drone_controller = DroneController()

    rclpy.spin(drone_controller)

    drone_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()