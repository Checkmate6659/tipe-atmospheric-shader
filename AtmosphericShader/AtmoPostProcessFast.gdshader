shader_type spatial;
render_mode unshaded; //important, otherwise sun reflects on the thing

//constant pi
const float pi = 3.14159265358979323846;

uniform float SCALING_FACTOR = 6371000.0; //conversion from scaled distance units to meters

uniform float R = 1.0; //planet radius (scaled distance unit)
uniform float H = 1.2e-3; //scale height (7.8km for earth)
uniform float NEGLIGIBLE_H = 6.0; //why is increasing the 10.0 breaking everything? (30: no atmo; 5: white)
uniform float BRIGHTNESS_MULTIPLIER = 1.0; //divided by 10^6 at the end, editor doesn't support tiny values well

const vec3 COLORS = vec3(612e-9, 549e-9, 464e-9); //screen wavelengths in meters
const vec3 SOLAR_IRRADIANCE = vec3(1.25e9, 1.3e9, 1.2e9); //solar irradiance in W*m^-3


//grrrr godot 4
uniform sampler2D SCREEN_TEXTURE : hint_screen_texture, filter_linear_mipmap;
uniform sampler2D DEPTH_TEXTURE : hint_depth_texture, filter_linear_mipmap;


//parameters
#include "params.gdshaderinc"


void vertex() {
  POSITION = vec4(2.0 * UV - vec2(1.0), 1.0, 1.0); //cover whole screen
}

//Calculate altitude
float altitude(vec3 position)
{
  return length(position) - R;
}

//project v orthogonally on plane normal to N going through origin
vec3 project(vec3 v, vec3 N)
{
  return v - dot(v, N)*N / dot(N, N);
}


//transition function
float leaky_relu(float x, float negative_slope)
{
    if (x > 0.) return x;
    return negative_slope * x;
}




//the whole light intensity function can be reduced to 4 params + lambda
//it is a good idea to use a spherical coord system with the main axis being earth-sun axis
//then camera longitude is eliminated by symmetry
//-height (altitude) of camera position "projected" into atmo
//-"latitude" of projected camera pos (<0 means far from sun, >0 means close to sun)
//-angle with which the camera is looking down (>0 means towards surface, <0 means towards sky)
//-angle with which the camera is looking to the side (can be picked >0 by symmetry too)
vec3 light_intensity_4param(vec3 lambda, float init_height, float init_lat, float down_angle, float side_angle)
{
  //calculate normalized parameters
  float l0[4] = { //layer 0
    exp(-init_height / H),
    init_lat / pi + 0.5,
    down_angle / pi + 0.5,
    side_angle / pi
  };


  float l1[16];
  float l2[16];
  float l3[3];

  //Initialize/bias l1, l2 and l3
  for (int i = 0; i < 16; i++)
  {
    l1[i] = BIASES1[i];
    l2[i] = BIASES2[i];
  }
  for (int i = 0; i < 3; i++)
    l3[i] = BIASES3[i];


  //First linear layer: l0 -> l1
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 4; j++)
      l1[i] += l0[j] * WEIGHTS1[4 * i + j];

  //Activation function of l1
  for(int i = 0; i < 16; i++) l1[i] = tanh(l1[i]);

  //Second linear layer: l1 -> l2
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      l2[i] += l1[j] * WEIGHTS2[16 * i + j];

  //Activation function of l2
  for(int i = 0; i < 16; i++) l2[i] = leaky_relu(l2[i], 0.0625);

  //Third linear layer: l2 -> l3
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 16; j++)
      l3[i] += l2[j] * WEIGHTS3[16 * i + j];


  return vec3(l3[0], l3[1], l3[2]);
}

//Compute the 4 necessary parameters for P_a and P_b
vec3 normalize_params(vec3 P_a, vec3 P_b, vec3 lambda)
{
  //calculate altitude of cam
  float init_height = altitude(P_a);

  //clamp points to the atmo's surface
  float Rp = R + NEGLIGIBLE_H*H; //10H is far enough from ground to neglect absorbance; 5H should be enough
  //calculate solutions where straight line hits sphere
  vec3 OP = P_a, v = P_b - P_a; //endpoint of the ray, and ray vector
  float one_over_v2 = 1.0 / dot(v, v); //get v/||v²||
  //now solve quadratic X²v² + 2Xv.OP + OP² - R² = 0
  float mid = -dot(v, OP) * one_over_v2; //calculate middle of 2 points
  float delta = (dot(v, OP) * dot(v, OP) * one_over_v2 + Rp*Rp - dot(OP, OP)) * one_over_v2; //discriminant (divide out by 4v²)
  float sqrt_delta = sqrt(max(delta, 0.0)); //careful about float errors!
  //calculate line abscissas of intersections
  float s1 = mid - sqrt_delta;
  float s2 = mid + sqrt_delta;
  //adjust start and end points to those; clamp them to be in line
  P_a = OP + clamp(s1, 0.0, 1.0) * v; //closer to eye (initial P_a)
  P_b = OP + clamp(s2, 0.0, 1.0) * v;

  float P_a_len = length(P_a);

  //calculate angles
  //TODO: try using sin or cos of the angles instead of calling asin and acos
  float init_lat = asin(P_a.z / P_a_len);
  float down_angle = asin(-dot(P_a, v) / (P_a_len * length(v)));

  //project vectors on plane normal to OP to find side angle
  vec3 vp = project(v, P_a), zp = project(vec3(0., 0., 1.), P_a);
  float side_angle = acos(clamp(dot(vp, zp) / (length(vp) * length(zp)), -1., 1.));

  // if(init_lat < 0.) return vec3(255., 0., 0.);
  return light_intensity_4param(lambda, init_height, init_lat, down_angle, side_angle);
}



//https://forum.godotengine.org/t/how-to-in-shader-get-the-pixels-3d-coordinate-local-and-global/20198/2
//use this to get ray vector, and do many things with it!!
void fragment() {
  //calculate depth
  vec3 col = texture(SCREEN_TEXTURE, SCREEN_UV).xyz;
  float depth = texture(DEPTH_TEXTURE, SCREEN_UV).x;
  vec3 ndc = vec3(SCREEN_UV, depth) * 2.0 - 1.0;
  vec4 view = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
  view.xyz /= view.w;
  float linear_depth = -view.z;

  ALBEDO = col;


  //camera position (world space)
  vec3 P = INV_VIEW_MATRIX[3].xyz;

  //v is a directing vector of a ray we are viewing
  //NOTE: for some reason this gets messed with by the far clipping plane, so max it out!
  //ALSO NOTE: we are assuming that we look at the planet, there is a second atmosphere rendering if we look opposite to it
  vec4 v4 = INV_VIEW_MATRIX * INV_PROJECTION_MATRIX * vec4((SCREEN_UV * 2.0) - 1.0, 0.0, 1.0);
  vec3 v = v4.xyz; //NOT a unit vector, quite close tho
  v /= length(v); //now it's a unit vector
  float vdotP = dot(v, P);
  float dsq = dot(P, P) - vdotP*vdotP; //can be negative due to float errors!

  float d; //closest distance to core at the end
  if(vdotP > 0.0) d = length(P); //going away from the core: camera position
  else d = sqrt(max(dsq, 0.0)); //shortest distance to core

  // ALBEDO = v; //TEST

  if(d < R + NEGLIGIBLE_H*H) //optimization: don't sim when going way too far from earth
  {
  	vec3 flux = normalize_params(P, P + v * 100.0, COLORS) * SOLAR_IRRADIANCE * COLORS; //irradiance in W/m^2
    ALBEDO = flux * BRIGHTNESS_MULTIPLIER * 1e-6;

    //add some random to avoid color banding
    //*should* have minimal performance impact
    float prng = fract(sin(dot(SCREEN_UV, vec2(12.9898,78.233))) * 43758.5453) - 0.5;
    ALBEDO += vec3(prng, prng, prng) / 1024.0;

    bool looking_at_sun = dot(v, vec3(0.0, 0.0, 1.0)) > 0.9999 && d > R;
    ALBEDO = float(looking_at_sun) * vec3(1.0, 1.0, 1.0) + float(!looking_at_sun) * ALBEDO;

    //if (d > R + NEGLIGIBLE_H*H && dot(ALBEDO, vec3(1.0)) > 0.03) ALBEDO = vec3(1.0, 0.0, 0.0);


    // ALBEDO = vec3(texture(TABLE, ndc.xy).xyz) / 256. * TABLE_MAX;
  }
  if(d < R) ALBEDO += col;

  // float mean = (ALBEDO.x + ALBEDO.y + ALBEDO.z) * 0.333;
  // ALBEDO *= 1.0 - exp(-42.0 * mean);

  // vec3 result = texelFetch(TABLE, ivec2(SCREEN_UV * 64.*64.), 0).xyz;
  // ALBEDO = result;


  // ALBEDO = light_intensity_4param(COLORS, H * log(2), (UV.y - 0.5) * pi, 0, UV.x * pi) * 1e-3;
}
