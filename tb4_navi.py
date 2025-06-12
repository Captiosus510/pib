import rclpy
from rclpy.node import Node
import time
from .openai_interface import PibTTSAgent
from openai import OpenAI
import json, cv2

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from action_msgs.msg import GoalStatus

class TalkNode(Node):
    system_prompt = """

        You are a helpful and concise humanoid robot that is in charge of controlling a robot. Respond in a friendly and natural tone. Don't use any emojis or special characters in your responses. You will be communicating through voice.

        You have a couple tools available to you in order to carry out the tasks effectively. Use them when necessary, but do not use them if they are not needed. The tools available to you are:

        1. take_picture: This tool allows you to take a picture from the camera feed. You can use this tool to capture images of your surroundings and analyze them. ONLY DO THIS WHEN NEEDED
        2. get_weather: This tool allows you to get the current weather for a specified location. You can use this tool to provide weather information to the user.
        3. check_dock: This tool allows you to check if the robot is docked. You can use this tool to determine if the robot is currently docked or not.
        4. dock: This tool allows you to dock the robot. You can use this tool to dock the robot when needed.
        5. undock: This tool allows you to undock the robot. You can use this tool to undock the robot when needed.

        Before calling a tool, always check whether it's necessary (if a check function exists). If a tool returns useful information, think about whether another action is required based on that result.

        For example, if the user says "dock", and the robot is already docked, do not call the docking function again. First check the docking status with check_dock, then only dock if needed.

        """
    tools = [
        {
            "type": "function",
            "name": "take_picture",
            "description": "Captures a picture from the camera feed.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_weather",
            "description": "Gets the current weather for a specified location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location for which to get the weather."
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "check_dock",
            "description": "Checks if the robot is docked.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "dock",
            "description": "Docks the robot.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "undock",
            "description": "Undocks the robot.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    ]

    cap = cv2.VideoCapture(0)

    max_messages = 20
    def __init__(self):
        super().__init__('talk_node')
        self.tts_agent = PibTTSAgent()
        self.client = OpenAI()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.timer = self.create_timer(1.0, self.prompt_user)
        self.busy = False
        self.navigator = TurtleBot4Navigator()
    
    def prompt_user(self):
        if not self.busy:
            self.busy = True
            self.get_logger().info('Please enter your command: ')
            user_input = input()
            try:
                if user_input.lower() in ["exit", "quit"]:
                    self.destroy_node()
                    rclpy.shutdown()
                    return

                self.add_message("user", user_input)
                response = self.get_response()
                self.get_logger().info(f"Response: {response}")
                while self.tts_agent.is_running():
                    time.sleep(0.1)
                self.busy = False
            except Exception as e:
                self.get_logger().error(f"Error in loop: {e}")
                self.busy = False
        else:
            self.get_logger().info('Still processing previous command...')
    
    def take_picture(self):
        status, frame = self.cap.read()
        if not status:
            raise Exception("Could not read frame from video device")
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        filename = "captured_image.jpg"
        cv2.imwrite(filename, frame)
        print("Picture taken and saved as", filename)
        image_file = self.client.files.create(file=open(filename, "rb"), purpose="vision")
        return image_file

    def get_weather(self, location):
        # Placeholder for weather retrieval logic
        return f"Current weather in {location} is sunny with a temperature of 25°C."
    
    def check_dock(self):
        return self.navigator.getDockedStatus()

    def dock(self):
        try:
            result = self.navigator.dock()
            return "Success"  # Assuming dock always succeeds for simplicity
            # if result == GoalStatus.STATUS_SUCCEEDED:
            #     return "Success"
            # else:
            #     return "Failed"
        except Exception as e:
            self.get_logger().error(f"Error docking: {e}")
            return "Error"

    def undock(self):
        try:
            result = self.navigator.undock()
            return "Success"  # Assuming undock always succeeds for simplicity
            # if result == GoalStatus.STATUS_SUCCEEDED:
            #     return "Success"
            # else:
            #     return "Failed"
        except Exception as e:
            self.get_logger().error(f"Error undocking: {e}")
            return "Error"

    
    def handle_tool_call(self, tool_call):
        name = tool_call.name
        args = json.loads(tool_call.arguments)

        if name == "take_picture":
            image_file = self.take_picture()
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "file_id": image_file.id,
                    }
                ]
            })
        elif name == "get_weather":
            self.messages.append(tool_call)
            weather_info = self.get_weather(**args)
            self.messages.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": str(weather_info)})
        elif name == "check_dock":
            self.messages.append(tool_call)
            docked_status = self.check_dock()
            self.messages.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": str(docked_status)})
        elif name == "dock":
            self.messages.append(tool_call)
            dock_result = self.dock()
            self.messages.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": str(dock_result)})
        elif name == "undock":
            self.messages.append(tool_call)
            undock_result = self.undock()
            self.messages.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": str(undock_result)})
        else:
            raise ValueError(f"Unknown function: {name}")
        
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.prune_messages(self.messages, self.max_messages)

    def prune_messages(self, messages, max_length):
        self.messages = [{"role": "system", "content": self.system_prompt}] + messages[-max_length:] if len(messages) > max_length else messages

    def get_response(self):
        stream = self.client.responses.create(
            model="gpt-4o",
            input=self.messages,
            tools=self.tools,
            tool_choice="auto",
            stream=True
        )

        buffer = ""
        text_out = ""
        tool_calls = []
        # streaming so text can be handled faster
        for event in stream:
            if event.type == "response.output_text.delta":
                self.tts_agent.start_thread_if_needed()
                token = event.delta
                buffer += token
                text_out += token
                # print(token, end="", flush=True)
                if token in [".", "!", "?"]:
                    self.tts_agent.speak(buffer, flush=False)
                    buffer = ""
            elif event.type == "response.output_item.done":
                if event.item.type == "function_call":
                    tool_calls.append(event.item)
                    print(f"\nTool call detected: {event.item.name} with args {json.loads(event.item.arguments)}")
                elif event.item.type == "message":
                    # text_out = event.item.content[0].text
                    # print(event.item)
                    self.messages.append({
                        "role": "assistant",
                        "content": text_out
                    })
                    
        # Flush any leftover text
        if buffer.strip():
            self.tts_agent.speak(buffer, flush=False)
        
        # handle tool calls if there are any
        if tool_calls:
            for tool_call in tool_calls:
                self.handle_tool_call(tool_call)
            return self.get_response()
        else:
            return text_out
        



def main():
    rclpy.init()
    node = TalkNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
