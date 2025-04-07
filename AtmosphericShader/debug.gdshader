shader_type spatial;
render_mode unshaded; //important, otherwise sun reflects on the thing


void vertex() {
  //worked in godot 3, not anymore
  //POSITION = vec4(VERTEX, 1.0); //cover whole screen

  //does nothing (on purpose)
  //vec4 vert = vec4(VERTEX, 1.0);
  //vert = MODELVIEW_MATRIX * vert;
  //vert = PROJECTION_MATRIX * vert;
  //POSITION = vert;

  POSITION = vec4(2.0 * UV - vec2(1.0), 1.0, 1.0); //why this works? idk
}

void fragment() {
  ALBEDO = vec3(SCREEN_UV, 0.0);
}
