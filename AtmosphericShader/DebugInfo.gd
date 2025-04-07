extends RichTextLabel

var debug_on = true
var fast_shader = false

func _input(event):
	# print(event.as_text())
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_P:
			debug_on = !debug_on
		if event.keycode == KEY_M: #switch shaders
			fast_shader = !fast_shader
			$"../Camera3D/BasicShader".visible = !fast_shader
			$"../Camera3D/FastShader".visible = fast_shader

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	if(debug_on):
		text = "Shader: " + ("FAST" if fast_shader else "BASIC") + \
		"\ndelta_t = " + str(1000 / Engine.get_frames_per_second()) + "ms\n" + \
		"altitude = " + str(($"../Camera3D".position.length() - 1) * 6371) + "km"
	else:
		text = ""

