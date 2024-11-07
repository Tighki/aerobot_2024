import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image, LaserScan
import cv2
import cv_bridge
import numpy as np

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        
        # Подписки на сообщения
        self.state_sub = self.create_subscription(State, '/uav1/mavros/state', self.state_cb, 10)
        self.local_pos_pub = self.create_publisher(PoseStamped, '/uav1/mavros/setpoint_position/local', 10)
        self.image_sub = self.create_subscription(Image, '/uav1/camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/uav1/laser/scan', self.laser_callback, 10)

        # Клиенты для арминга и режима
        self.arming_client = self.create_client(CommandBool, '/uav1/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/uav1/mavros/set_mode')

        # Инициализация
        self.current_state = State()
        self.pose = PoseStamped()
        self.cv_bridge = cv_bridge.CvBridge()
        
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
            self.get_logger().info("Drone is taking off to search for QR codes...")

        if self.search_mode:
            if self.is_obstacle_ahead():
                self.avoid_obstacle()
            else:
                self.explore_environment()
        
        self.local_pos_pub.publish(self.pose)

    def is_obstacle_ahead(self):
        # Используем данные лазера для проверки препятствий
        filtered_laser_data = self.filter_laser_data(self.laser_data)
        return any(distance < self.safe_distance for distance in filtered_laser_data)

    def avoid_obstacle(self):
        self.get_logger().info("Obstacle detected! Avoiding...")
        filtered_laser_data = self.filter_laser_data(self.laser_data)
        min_distance = min(filtered_laser_data)
        min_index = filtered_laser_data.index(min_distance)

        # Определяем направление обхода на основе расположения минимального расстояния
        if min_index < len(filtered_laser_data) // 3:
            self.pose.pose.position.y += 1  # Уход вправо
        elif min_index > 2 * len(filtered_laser_data) // 3:
            self.pose.pose.position.y -= 1  # Уход влево
        else:
            self.pose.pose.position.x -= 0.7  # Движение назад

        self.local_pos_pub.publish(self.pose)

    def filter_laser_data(self, laser_data):
        # Фильтрация лазерных данных: исключаем слишком большие расстояния или шумы
        return [distance for distance in laser_data if distance < self.safe_distance * 2 and distance > 0.2]

    def explore_environment(self):
        self.pose.pose.position.x += 0.1
        if self.pose.pose.position.x > 5:
            self.pose.pose.position.x = 2.0
            self.pose.pose.position.y += 0.5
            if self.pose.pose.position.y > 5:
                self.pose.pose.position.y = 3.0

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

    def image_callback(self, msg):
        # Получаем изображение и обрабатываем его с помощью OpenCV
        frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(frame)

        # Если QR-код найден
        if data:
            self.get_logger().info(f"QR Code Detected: {data}")
            self.navigate_to_qr_code(bbox)
            self.search_mode = False

        # Обработка препятствий с помощью OpenCV (детекция объектов)
        self.detect_obstacles(frame)

    def detect_obstacles(self, frame):
        # Применяем более сложные методы, например, контурную детекцию или фильтрацию объектов
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Фильтруем слишком маленькие объекты
                # Отметим найденные контуры
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.get_logger().info(f"Obstacle detected at x: {x}, y: {y}")
                self.avoid_obstacle()

        # Показать изображение (для отладки)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

    def navigate_to_qr_code(self, bbox):
        if bbox is not None:
            # Вычисляем координаты центра QR-кода
            center_x = (bbox[0][0][0] + bbox[2][0][0]) / 2
            center_y = (bbox[0][0][1] + bbox[2][0][1]) / 2
            
            offset_x = center_x - 320
            offset_y = center_y - 240
            
            self.pose.pose.position.x += offset_x * 0.001
            self.pose.pose.position.y += offset_y * 0.001
            self.pose.pose.position.z = self.altitude
            self.local_pos_pub.publish(self.pose)

def main(args=None):
    rclpy.init(args=args)
    drone_controller = DroneController()

    rclpy.spin(drone_controller)

    drone_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
