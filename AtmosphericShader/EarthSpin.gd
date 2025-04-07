extends MeshInstance3D

#@export var EARTH_SENSITIVITY = 0.03

#func _input(event):
#	if event.is_action("ui_right"):
#		$".".rotate_y(EARTH_SENSITIVITY);
#	if event.is_action("ui_left"):
#		$".".rotate_y(-EARTH_SENSITIVITY);

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	$".".rotate_y(delta / (2*PI))

