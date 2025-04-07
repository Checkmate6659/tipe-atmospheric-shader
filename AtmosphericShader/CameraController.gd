extends Camera3D

@export var mouse_sens = 0.01

var horiz_angle = deg_to_rad(-90)
var vert_angle = 0

# var posOffset = Vector3.ZERO
var posOffset = Vector3(0, 0, 3)

func _process(_delta):
	if Input.is_key_pressed(KEY_SHIFT):
		_delta *= 0.05 #precise control
	if Input.is_key_pressed(KEY_CTRL):
		_delta *= 0.0025 #even more precise control
	if Input.is_key_pressed(KEY_Z):
		posOffset -= get_global_transform().basis.z * _delta
	elif Input.is_key_pressed(KEY_S):
		posOffset += get_global_transform().basis.z * _delta
	elif Input.is_key_pressed(KEY_Q):
		posOffset -= get_global_transform().basis.x * _delta
	elif Input.is_key_pressed(KEY_D):
		posOffset += get_global_transform().basis.x * _delta
	elif Input.is_key_pressed(KEY_A):
		posOffset -= get_global_transform().basis.y * _delta
	elif Input.is_key_pressed(KEY_E):
		posOffset += get_global_transform().basis.y * _delta

	#move camera
	position = $"../MeshInstance3D".position + posOffset
	
	#update the rotation (mb a bit wacky)
	look_at(Vector3(cos(horiz_angle) * cos(vert_angle), sin(vert_angle), sin(horiz_angle) * cos(vert_angle)) + $"../MeshInstance3D".position + posOffset, Vector3(0, 1, 0))

func _input(event):
	if event is InputEventMouseMotion:
		if Input.is_mouse_button_pressed(1): #gives a warning because 1 is not an enum and the thing needs an enum
			var sens_modifier = 1
			if Input.is_key_pressed(KEY_SHIFT):
				sens_modifier = 0.01 #precise control

			#update angles
			horiz_angle += event.relative.x*mouse_sens*sens_modifier
			vert_angle -= event.relative.y*mouse_sens*sens_modifier
			vert_angle = clamp(vert_angle, deg_to_rad(-90), deg_to_rad(90))
