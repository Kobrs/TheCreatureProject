import ev3

# define commands
class Brick(object):
	def __init__(self, port, motor_time):
		self.motor_time = motor_time
		self.port = port

		self.get_LR_sensors = ev3.direct_command.DirectCommand()
		self.get_LR_sensors.add_input_device_ready_raw(
								ev3.direct_command.InputPort.PORT_4,
								ev3.direct_command.ColorMode.AMBIENT,
								ev3.direct_command.DeviceType.EV3_COLOR)
		self.get_LR_sensors.add_input_device_ready_raw(
								ev3.direct_command.InputPort.PORT_1,
								ev3.direct_command.ColorMode.AMBIENT,
								ev3.direct_command.DeviceType.EV3_COLOR)


	def move_tracks(self, Lspeed, Rspeed):
		# Ltrack_cmd = ev3.direct_command.DirectCommand()
		# Ltrack_cmd.add_output_speed(ev3.direct_command.OutputPort.PORT_D, motor_speed)
		# Ltrack_cmd.add_output_start(ev3.direct_command.OutputPort.PORT_D)
		# Ltrack_cmd.add_timer_wait(self.motor_time)
		# Ltrack_cmd.add_output_stop(ev3.direct_command.OutputPort.PORT_D,
		# 						   ev3.direct_command.StopType.BRAKE)
		# Ltrack_cmd.send(self.brick)


		cmd = ev3.direct_command.DirectCommand()
		cmd.add_output_speed(ev3.direct_command.OutputPort.PORT_D, Lspeed)
		cmd.add_output_speed(ev3.direct_command.OutputPort.PORT_A, Rspeed)
		cmd.add_output_start(ev3.direct_command.OutputPort.PORT_D)
		cmd.add_output_start(ev3.direct_command.OutputPort.PORT_A)
		cmd.add_timer_wait(self.motor_time)
		cmd.add_output_stop(ev3.direct_command.OutputPort.PORT_D,
								   ev3.direct_command.StopType.BRAKE)
		cmd.add_output_stop(ev3.direct_command.OutputPort.PORT_A,
								   ev3.direct_command.StopType.BRAKE)
		
		with ev3.ev3.EV3(port_str=self.port) as brick:
			cmd.send(brick)



	def get_sensors_state(self):
		# TODO: maybe I should split command creation and command running so that
		# 		this class creates commands but whit statement with brick
		# 		intit is in main file. This wouldn't lead to so many connections
		# 		in short times.
		with ev3.ev3.EV3(port_str=self.port) as brick:
			# L = self.get_left_sensor.send(brick)
			# R = self.get_right_sensor.send(brick)
			LR = self.get_LR_sensors.send(brick)
		return LR

		

	def move_tracks_cmd(self, Lspeed, Rspeed):
		# Ltrack_cmd = ev3.direct_command.DirectCommand()
		# Ltrack_cmd.add_output_speed(ev3.direct_command.OutputPort.PORT_D, motor_speed)
		# Ltrack_cmd.add_output_start(ev3.direct_command.OutputPort.PORT_D)
		# Ltrack_cmd.add_timer_wait(self.motor_time)
		# Ltrack_cmd.add_output_stop(ev3.direct_command.OutputPort.PORT_D,
		# 						   ev3.direct_command.StopType.BRAKE)
		# Ltrack_cmd.send(self.brick)


		cmd = ev3.direct_command.DirectCommand()
		cmd.add_output_speed(ev3.direct_command.OutputPort.PORT_D, Lspeed)
		cmd.add_output_speed(ev3.direct_command.OutputPort.PORT_A, Rspeed)
		cmd.add_output_start(ev3.direct_command.OutputPort.PORT_D)
		cmd.add_output_start(ev3.direct_command.OutputPort.PORT_A)
		cmd.add_timer_wait(self.motor_time)
		cmd.add_output_stop(ev3.direct_command.OutputPort.PORT_D,
								   ev3.direct_command.StopType.BRAKE)
		cmd.add_output_stop(ev3.direct_command.OutputPort.PORT_A,
								   ev3.direct_command.StopType.BRAKE)
		
		# with ev3.ev3.EV3(port_str=self.port) as brick:
		# 	cmd.send(brick)
		return cmd



	def get_sensors_state_cmd(self):
		# TODO: maybe I should split command creation and command running so that
		# 		this class creates commands but whit statement with brick
		# 		intit is in main file. This wouldn't lead to so many connections
		# 		in short times.
		# with ev3.ev3.EV3(port_str=self.port) as brick:
		# 	# L = self.get_left_sensor.send(brick)
		# 	# R = self.get_right_sensor.send(brick)
		# 	LR = self.get_LR_sensors.send(brick)
		# return LR
		return self.get_LR_sensors

		