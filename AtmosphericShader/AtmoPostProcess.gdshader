shader_type spatial;
render_mode unshaded; //important, otherwise sun reflects on the thing

//constant pi
const float pi = 3.14159265358979323846;

uniform float SCALING_FACTOR = 6371000.0; //conversion from scaled distance units to meters

uniform float R = 1.0; //planet radius (scaled distance unit)
uniform float H = 1.2e-3; //scale height (7.8km for earth)
//https://www.brown.edu/research/labs/mittleman/sites/brown.edu.research.labs.mittleman/files/uploads/lecture08_0.pdf
//page 7: there is a formula for alpha, but i don't understand all the terms
uniform float ALPHA_0 = 6371.0; //absorbance at sea level; value is 1km^-1 ie 1e-3 m^-1
uniform float OPT_INDEX = 1.00029; //air optical index (changes with color, but negligible)
uniform float NUMBER_DENSITY = 2.504e25; //molecular number density in m^-3
uniform float NEGLIGIBLE_H = 6.0; //why is increasing the 10.0 breaking everything? (30: no atmo; 5: white)
uniform float BRIGHTNESS_MULTIPLIER = 1.0; //divided by 10^6 at the end, editor doesn't support tiny values well

const vec3 COLORS = vec3(612e-9, 549e-9, 464e-9); //screen wavelengths in meters
const vec3 SOLAR_IRRADIANCE = vec3(1.25e9, 1.3e9, 1.2e9); //solar irradiance in W*m^-3

uniform int TAU_POINTS = 50; //points for integration when calculating optical depth
uniform int I_POINTS = 50; //points for integration when calculating light intensity


//grrrr godot 4
uniform sampler2D SCREEN_TEXTURE : hint_screen_texture, filter_linear_mipmap;
uniform sampler2D DEPTH_TEXTURE : hint_depth_texture, filter_linear_mipmap;


void vertex() {
  POSITION = vec4(2.0 * UV - vec2(1.0), 1.0, 1.0); //cover whole screen
}

//Calculate altitude
float altitude(vec3 position)
{
  return sqrt(dot(position, position)) - R;
}

//Calculate K/lambda^4 value
vec3 K_over_lambda4(vec3 lambda)
{
  float thing = (OPT_INDEX*OPT_INDEX - 1.0); //don't recalculate it twice
  vec3 lambda2 = lambda*lambda; //same here

  //http://nishitalab.org/user/nis/cdrom/sig93_nis.pdf
  return 2.0*pi*pi * thing*thing / (3.0*NUMBER_DENSITY*lambda2*lambda2);
}

//Calculate absorbance at a certain position
float absorbance(vec3 position)
{
  return ALPHA_0 * exp(-altitude(position) / H);
}

//Calculate optical depth: integrate absorbance over P_a and P_b positions
//Uses trapezoidal approximation
float optical_depth(vec3 P_a, vec3 P_b)
{
  //get points as close to surface of the sphere as possible
  float Rp = R + NEGLIGIBLE_H*H; //10H is far enough from ground to neglect absorbance; 5H should be enough
  //calculate solutions where straight line hits sphere
  vec3 OP = P_a, v = P_b - P_a;
  float one_over_v2 = 1.0 / dot(v, v); //get v/||v²||
  //now solve quadratic X²v² + 2Xv.OP + OP² - R² = 0
  float mid = -dot(v, OP) * one_over_v2; //calculate middle of 2 points
  float delta = (dot(v, OP) * dot(v, OP) * one_over_v2 + Rp*Rp - dot(OP, OP)) * one_over_v2; //discriminant (divide out by 4v²)
  float sqrt_delta = sqrt(max(delta, 0.0)); //careful about float errors!
  //calculate line abscissas of intersections
  float s1 = mid - sqrt_delta;
  float s2 = mid + sqrt_delta;
  //adjust start and end points to those; clamp them to be in line
  P_a = OP + clamp(s1, 0.0, 1.0) * v; //closer to first P_a
  P_b = OP + clamp(s2, 0.0, 1.0) * v;

  float sum = (absorbance(P_a) + absorbance(P_b)) * 0.5;
  for(int i = 1; i < TAU_POINTS; i++)
  {
    float ratio = float(i) / float(TAU_POINTS);
    vec3 position = P_a * ratio + P_b * (1.0 - ratio);
    if (altitude(position) < 0.) return 1./0.; //can't go through! infinite absorption
    sum += absorbance(position);
  }

  return sum * sqrt(dot(P_b - P_a, P_b - P_a)) * SCALING_FACTOR / float(TAU_POINTS);
}

//Calculate t-value between 2 positions
vec3 t(vec3 P_a, vec3 P_b, vec3 lambda)
{
  return 4.0*pi*K_over_lambda4(lambda) * optical_depth(P_a, P_b) / ALPHA_0;
}

//phase function; set g = 0 for rayleigh scattering
float F(float costheta, float g)
{
  float g2 = g*g;
  float cos2theta = costheta * costheta;
  return 1.5 * (1.0 - g2) * (1.0 + cos2theta) /
    ((2.0 + g2) * pow(1.0 + g2 - 2.0*g*costheta, 1.5));
}

//Calculate a bit of outgoing intensity using an integral
//Uses trapezoidal approximation
//Also, you still need to multiply by K/lambda^4, I_s and F(theta, g) to get I_v
//NOTE: it may be possible that earth isn't actually *blocking* light
//which means atmo bright even when sun isn't visible!
vec3 light_intensity_multiplier(vec3 P_a, vec3 P_b, vec3 lambda)
{
  //get points as close to surface of the sphere as possible
  //when Rp is increased, NO ATMO!
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

  //if going under the surface, get P_b right on top of the surface
  vec3 midpoint = OP + mid * v;
  if(mid > 0.0 && dot(midpoint, midpoint) < R*R) //under the surface
  {
    //put P_b at the closest point to the surface
    float delta2 = (dot(v, OP) * dot(v, OP) * one_over_v2 + R*R - dot(OP, OP)) * one_over_v2;
    P_b = OP + v * (mid - sqrt(max(delta2, 0.0)));
  }

  vec3 sum = (absorbance(P_a) * exp(-t(P_a, P_a + vec3(0.0, 0.0, Rp*2.0), lambda))
    + absorbance(P_b) * exp(-t(P_b, P_a, lambda) - t(P_b, P_b + vec3(0.0, 0.0, Rp*2.0), lambda))
  ) * 0.5;
  for(int i = 1; i < I_POINTS; i++)
  {
    float ratio = float(i) / float(I_POINTS);
    vec3 P = P_a * ratio + P_b * (1.0 - ratio);

    if (altitude(P) < 0.) //stop counting the points past surface!
      break;

    //sun is in positive Z direction; evaluate integrand
    sum += absorbance(P) * exp(-t(P, P_a, lambda) - t(P, P + vec3(0.0, 0.0, Rp*2.0), lambda));
  }

  return sum * sqrt(dot(P_b - P_a, P_b - P_a)) * SCALING_FACTOR / float(I_POINTS); //multiply by ds
}

//calculate the full light intensity, but not multiplied by solar intensity so you need to do that still
vec3 light_intensity(vec3 P_a, vec3 P_b, vec3 lambda)
{
  vec3 multiplier = light_intensity_multiplier(P_a, P_b, lambda);

  //cos(theta) = -u.v/||v|| where u is sunlight's directing unit vector,
  //and v is any vector directing the ray (here P_b - P_a)
  vec3 v = P_b - P_a, u = vec3(0.0, 0.0, -1.0);
  float costheta = -dot(u, v) * pow(dot(v, v), -0.5); //no rsqrt function?

  return multiplier * F(costheta, 0.0) * K_over_lambda4(lambda);
}

//https://forum.godotengine.org/t/how-to-in-shader-get-the-pixels-3d-coordinate-local-and-global/20198/2
//use this to get ray vector, and do many things with it!!
void fragment() {
  //calculate depth
  vec3 col = texture(SCREEN_TEXTURE, SCREEN_UV).xyz;
  float depth = texture(DEPTH_TEXTURE, SCREEN_UV).x;
  vec3 ndc = vec3(UV, depth) * 2.0 - 1.0;
  vec4 view = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
  view.xyz /= view.w;
  float linear_depth = -view.z;

  ALBEDO = col;

  //distance of straight line to point:
  //assume P is a point, and v is a vector
  //we want distance d of origin to line going through P and directed by v
  //let theta be the angle between the 2
  //cos²(theta) = (P.v)²/[|P|²|v|²]
  //d = |P| sin theta = |P| sqrt(1 - cos² theta)
  //d = sqrt(P² - (v.P)²/v²)

  //vec4 world = CAMERA_MATRIX * INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
  //vec3 world_space = world.xyz / world.w;

  //TODO: learn this
  //https://docs.godotengine.org/en/3.5/tutorials/shaders/shader_reference/spatial_shader.html#fragment-built-ins

  //camera position (world space)
  vec3 P = INV_VIEW_MATRIX[3].xyz;


  //v is a directing vector of a ray we are viewing
  //NOTE: for some reason this gets messed with by the far clipping plane, so max it out!
  //ALSO NOTE: we are assuming that we look at the planet, there is a second atmosphere rendering if we look opposite to it
  vec4 v4 = INV_VIEW_MATRIX * INV_PROJECTION_MATRIX * vec4(2.0 * UV - vec2(1.0), 0.0, 1.0);
  vec3 v = v4.xyz; //NOT a unit vector, quite close tho
  v /= length(v); //now it's a unit vector
  float vdotP = dot(v, P);
  float dsq = dot(P, P) - vdotP*vdotP; //can be negative due to float errors!

  float d; //closest distance to core at the end
  if(vdotP > 0.0) d = sqrt(dot(P, P)); //going away from the core: camera position
  else d = sqrt(max(dsq, 0.0)); //shortest distance to core

  // ALBEDO += v;


  if(d < R + NEGLIGIBLE_H*H) //optimization: don't sim when going way too far from earth
  {
    //show absorbance
    //ALBEDO = col + vec3(0.3, 0.5, 1.0) * exp((R-d)*100.0);

    //show optical depth
    //sun is in positive Z direction
    //ALBEDO = col + vec3(0.3, 0.5, 1.0) * optical_depth(P, P + v * 5.0) / SCALING_FACTOR;

    //ALBEDO = col + t(P, P + v * 5.0, COLORS);

    //calculate final irradiance/flux density
  	vec3 flux = light_intensity(P, P + v * 9999., COLORS) * SOLAR_IRRADIANCE * COLORS; //irradiance in W/m^2
    ALBEDO = flux * BRIGHTNESS_MULTIPLIER * 1e-6;

    //add some random to avoid color banding
    //*should* have minimal performance impact
    float prng = fract(sin(dot(SCREEN_UV, vec2(12.9898,78.233))) * 43758.5453) - 0.5;
    ALBEDO += vec3(prng, prng, prng) / 1024.0;

    bool looking_at_sun = dot(v, vec3(0.0, 0.0, 1.0)) > 0.9999 && d > R;
    ALBEDO = float(looking_at_sun) * vec3(1.0, 1.0, 1.0) + float(!looking_at_sun) * ALBEDO;

    //if (d > R + NEGLIGIBLE_H*H && dot(ALBEDO, vec3(1.0)) > 0.03) ALBEDO = vec3(1.0, 0.0, 0.0);
  }
  if(d < R) ALBEDO += col;


  // if(d < R)
    // ALBEDO = vec3(255., 0., 0.);
  // float mean = (ALBEDO.x + ALBEDO.y + ALBEDO.z) * 0.333;
  // ALBEDO *= 1.0 - exp(-42.0 * mean);
}
