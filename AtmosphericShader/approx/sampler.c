#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

typedef struct {
    double x, y, z;
} vec3;

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SCALING_FACTOR 6371000.0 //conversion from scaled distance units to meters
#define R 1.0 //planet radius (scaled distance unit)
#define H 1.2e-3 //scale height (7.8km for earth)
#define ALPHA_0 6371.0 //absorbance at sea level: 1e-3 m^-1
#define OPT_INDEX 1.00029 //air optical index at sea level (changes with color, but negligible)
#define NUMBER_DENSITY 2.504e25 //molecular number density at sea level in m^-3
#define NEGLIGIBLE_H 10.0

const vec3 COLORS = (vec3){612e-9, 549e-9, 464e-9}; //screen wavelengths in meters
const vec3 IRRADIANCE = (vec3){1.25e9, 1.3e9, 1.2e9}; //solar irradiance in W*m^-3

const int TAU_POINTS = 8;//64; //points for integration when calculating optical depth (BUMP IT UP)
const int I_POINTS = 32; //128; //points for integration when calculating light intensity

double dot(vec3 v1, vec3 v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

vec3 cross(vec3 v1, vec3 v2)
{
    return (vec3){
        v1.y*v2.z - v2.y*v1.z,
        v1.z*v2.x - v2.z*v1.x,
        v1.x*v2.y - v2.x*v1.y
    };
}

double clamp(double x, double m, double M)
{
    return MIN(MAX(x, m), M);
}

//a + b
vec3 vec3_plus(vec3 a, vec3 b)
{
    return (vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

//a - b
vec3 vec3_minus(vec3 a, vec3 b)
{
    return (vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

//kv
vec3 vec3_scalarmul(double k, vec3 v)
{
    return (vec3){v.x*k, v.y*k, v.z*k};
}

//============== START OF REAL CODE ==============
double altitude(vec3 position)
{
    return sqrt(dot(position, position)) - R;
}

double K_over_lambda4(double lambda)
{
    const double thing = (OPT_INDEX*OPT_INDEX - 1.0); //don't recalculate it twice
    double lambda2 = lambda*lambda; //same here
    //http://nishitalab.org/user/nis/cdrom/sig93_nis.pdf
    return 2.0*M_PI*M_PI * thing*thing / (3.0*NUMBER_DENSITY*lambda2*lambda2);
}

double absorbance(vec3 position)
{
    return ALPHA_0 * exp(-altitude(position) / H);
}

double distance_of_segment_to_center(vec3 P_a, vec3 P_b)
{
    vec3 v = vec3_minus(P_b, P_a);
    double lensq = dot(v, v); //check if length is 0
    if(lensq == 0) return sqrt(dot(P_a, P_a));
    //parametrize segment by a variable t (between 0 and 1)
    //calculate length of projected vector and divide it
    //just done without square roots
    double t = MAX(0, MIN(1, -dot(P_a, v) / lensq));
    vec3 closest_point = vec3_plus(P_a, vec3_scalarmul(t, v));
    return sqrt(dot(closest_point, closest_point));
}

bool useless(vec3 P_a, vec3 P_b)
{
    //these points are gonna get mapped to on the surface of the sphere anyway => USELESS
    //this is good when we do the whole surface, less for visualization but make sure we turn it on
    if (dot(P_a, P_a) > (R + H * NEGLIGIBLE_H)*(R + H * NEGLIGIBLE_H) &&
        dot(P_b, P_b) > (R + H * NEGLIGIBLE_H)*(R + H * NEGLIGIBLE_H))
        return true;

    //goes through surface: USELESS
    double dist = distance_of_segment_to_center(P_a, P_b);
    return dist < R || dist > R + H * NEGLIGIBLE_H;
}

//Clamp vectors at P_a and P_b to inside ball of radius Rp
//in a way that the line segment between them and in the ball
//stays the same; returns true iff operation succeeded (P_a != P_b)
bool clamp_in_ball(vec3 *P_a, vec3 *P_b, double Rp)
{
    vec3 OP = *P_a;
    vec3 v = vec3_minus(*P_b, *P_a); //direction of ray
    if (dot(v, v) == 0) return false; //no distance
    double one_over_v2 = 1.0 / dot(v, v); //get 1/||v||² to get v/||v||²

    //now solve quadratic X²v² + 2Xv.OP + OP² - R² = 0
    //and we can parametrize the line segment, which is great!
    double mid = -dot(v, OP) * one_over_v2; //calculate middle of 2 points
    double delta = (dot(v, OP) * dot(v, OP) * one_over_v2 +
        Rp*Rp - dot(OP, OP)) * one_over_v2; //discriminant (divide out by 4v²)
    double sqrt_delta = sqrt(MAX(delta, 0.0)); //careful about double errors!

    //calculate line abscissas of intersections
    double s1 = mid - sqrt_delta;
    double s2 = mid + sqrt_delta;

    //adjust start and end points to those; clamp them to be in line
    *P_a = vec3_plus(OP, vec3_scalarmul(clamp(s1, 0.0, 1.0), v)); //closer to first P_a
    *P_b = vec3_plus(OP, vec3_scalarmul(clamp(s2, 0.0, 1.0), v));
    return true;
}

//calculate optical depth: integrate absorbance between P_a and P_b
//using trapezoidal approximation
double optical_depth(vec3 P_a, vec3 P_b)
{
    clamp_in_ball(&P_a, &P_b, R + NEGLIGIBLE_H*H); //don't integrate in space!

    double csum = (absorbance(P_a) + absorbance(P_b)) * 0.5;
    for (int i = 1; i < TAU_POINTS; i++)
    {
        double ratio = i / (double)TAU_POINTS;
        vec3 position = vec3_plus(vec3_scalarmul(ratio, P_a),
            vec3_scalarmul(1.0 - ratio, P_b));
        if (altitude(position) < 0.) return INFINITY; //can't go through! infinite absorption
        csum += absorbance(position);
    }

    vec3 v = vec3_minus(P_b, P_a);
    return csum * sqrt(dot(v, v)) * SCALING_FACTOR / TAU_POINTS; //multiply it all by dx
}

//Calculate t-value between 2 positions
double t(vec3 P_a, vec3 P_b, double lambda)
{
      return 4.0*M_PI*K_over_lambda4(lambda) * optical_depth(P_a, P_b) / ALPHA_0;
}

//phase function; set g = 0 for rayleigh scattering
double F(double costheta, double g)
{
    double g2 = g*g;
    double cos2theta = costheta * costheta;
    return 1.5 * (1.0 - g2) * (1.0 + cos2theta) /
        ((2.0 + g2) * pow(1.0 + g2 - 2.0*g*costheta, 1.5));
}


//Calculate a bit of outgoing intensity using an integral
//Uses trapezoidal approximation
//Also, you still need to multiply by K/lambda^4, I_s and F(theta, g) to get I_v
//NOTE: it may be possible that earth isn't actually *blocking* light
//which means atmo bright even when sun isn't visible!
double light_intensity_multiplier(vec3 P_a, vec3 P_b, double lambda)
{
    //get points as close to surface of the sphere as possible
    double Rp = R + NEGLIGIBLE_H*H; //10H is far enough from ground to neglect absorbance; 5H should be enough
    //calculate solutions where straight line hits sphere
    vec3 OP = P_a;
    vec3 v = vec3_minus(P_b, P_a);
    double one_over_v2 = 1.0 / dot(v, v); //get v/||v²||
    double mid = -dot(v, OP) * one_over_v2; //calculate middle of 2 intersection points
    
    clamp_in_ball(&P_a, &P_b, R + NEGLIGIBLE_H*H); //don't integrate in space!

    //if going under the surface, get P_b right on top of the surface
    vec3 midpoint = vec3_plus(OP, vec3_scalarmul(mid, v));
    if(mid > 0.0 && dot(midpoint, midpoint) < R*R) //under the surface
    {
        //put P_b at the closest point to the surface
        double delta2 = (dot(v, OP) * dot(v, OP) * one_over_v2 + R*R - dot(OP, OP)) * one_over_v2;
        P_b = vec3_plus(OP, vec3_scalarmul(mid - sqrt(MAX(delta2, 0.0)), v));
    }

    double csum = (absorbance(P_a) * exp(-t(P_a, vec3_plus(P_a, (vec3){0.0, 0.0, Rp*2.0}), lambda))
        + absorbance(P_b) * exp(-t(P_b, P_a, lambda) - t(P_b, vec3_plus(P_b, (vec3){0.0, 0.0, Rp*2.0}), lambda))
        ) * 0.5;

    for (int i = 1; i < I_POINTS; i++)
    {
        double ratio = i / (double)I_POINTS;
        vec3 P = vec3_plus(vec3_scalarmul(ratio, P_a), vec3_scalarmul(1.0 - ratio, P_b));

        //next line not necessary because infinite absorption -> exp is 0; just more calculation done
        if (altitude(P) < 0.)
            break; //stop counting the points past surface! (does it optimize?)

        //sun is in positive Z direction; evaluate integrand
        csum += absorbance(P) * exp(-t(P, P_a, lambda) - t(P, vec3_plus(P, (vec3){0.0, 0.0, Rp*2.0}), lambda));
    }

    v = vec3_minus(P_b, P_a);
    return csum * sqrt(dot(v, v)) * SCALING_FACTOR / (double)I_POINTS; //multiply by ds
}

//calculate the full light intensity, but not multiplied by solar intensity so you need to do that still
double light_intensity(vec3 P_a, vec3 P_b, double lambda)
{
    double multiplier = light_intensity_multiplier(P_a, P_b, lambda);

    //cos(theta) = -u.v/||v|| where u is sunlight's directing unit vector,
    //and v is any vector directing the ray (here P_b - P_a)
    vec3 v = vec3_minus(P_b, P_a);
    vec3 u = (vec3){0.0, 0.0, -1.0};
    double costheta = -dot(u, v) / sqrt(dot(v, v));

    return multiplier * F(costheta, 0.0) * K_over_lambda4(lambda);
}

//Return the vector v spun around the axis u by the angle angle
//The axis u must be a unit vector, v doesn't
//https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
//i don't understand where this formula comes from really
vec3 spin_vector(vec3 v, vec3 u, double angle)
{
    double c = cos(angle);
    double s = sin(angle);
    return vec3_plus(vec3_scalarmul(c, v),
    vec3_plus(vec3_scalarmul(s, cross(u, v)),
    vec3_scalarmul((1-c)*dot(u, v), u)));
}

//the whole light intensity function can be reduced to 4 params + lambda
//it is a good idea to use a spherical coord system with the main axis being earth-sun axis
//then camera longitude is eliminated by symmetry
//-height (altitude) of camera position "projected" into atmo
//-"latitude" of projected camera pos (<0 means far from sun, >0 means close to sun)
//-angle with which the camera is looking down (>0 means towards surface, <0 means towards sky)
//-angle with which the camera is looking to the side (can be picked >0 by symmetry too)
double light_intensity_4param(double lambda, double init_height, double init_lat, double down_angle, double side_angle)
{
    //sun in +Z direction, arbitrarily pick X coordinate to be 0
    vec3 ax = (vec3){0, cos(init_lat), sin(init_lat)};
    vec3 P_a = vec3_scalarmul(R + init_height, ax);

    //compute a directing vector of ray
    //first an intermediate vector
    vec3 v_0 = (vec3){0, -sin(init_lat + down_angle), cos(init_lat + down_angle)};
    //then rotate it to the side around the P_a axis
    vec3 v = spin_vector(v_0, ax, side_angle);

    //lazy method: just adding a "view vector" long enough
    //v is a unit vector
    vec3 P_b = vec3_plus(P_a, vec3_scalarmul(2 * (R + NEGLIGIBLE_H * H), v));

    //finally calculate light intensity
    double intensity = light_intensity(P_a, P_b, lambda);
    if (isnan(intensity)) return 0;
    return intensity;
}

//same as prev function, just all params are between 0 and 1 now (affine scaling)
double normalized_4param(double lambda, double init_height, double init_lat, double down_angle, double side_angle)
{
    double atmo_height = NEGLIGIBLE_H * H;
    return light_intensity_4param(lambda,
        // init_height * atmo_height, //[0; atmo height]
        MIN(-log(init_height) * H, atmo_height), //[0; atmo height]; a bit of transformation
        -M_PI/2 + init_lat * M_PI, //[-pi/2; pi/2]
        -M_PI/2 + down_angle * M_PI, //[-pi/2; pi/2]
        side_angle * M_PI //[0; pi]
    );
}

//SAMPLER ONLY CODE
#include <pthread.h>

//print mutex
pthread_mutex_t print_mutex;


//Uniform distribution on [0; 1]... pretty much.
double random_uni01()
{
    //windows => RAND_MAX = 32767, too low!
    double DRAND_MAX = (RAND_MAX + 1) * (RAND_MAX + 1) - 1;
    return (double)(rand() * (RAND_MAX + 1) + rand()) / DRAND_MAX;
}

//gaussian distribution, using box-muller transform
//mean = 0; sigma = 1
double random_gauss()
{
    double theta = random_uni01() * 2 * M_PI;
    double r = sqrt(-2 * log(random_uni01()));
    return r * cos(theta);
}

//Non-uniform sampling: the function is not interesting everywhere
//-Skew beta to mostly be contained in 1/20 length interval in center of the image
//-Skew alpha to mostly be contained in [0.4; 1], ie in daytime
void random_sample(double *h, double *a, double *b, double *g)
{
    //just uniformly sample those
    *h = random_uni01();
    *g = random_uni01();

    //non-uniform sampling of alpha
    if(random_uni01() < 0.4) //40% of all samples are in sunset strip
        *a = random_uni01() * 0.1 + 0.45;
    else if(random_uni01() < 0.7) //70% probability of day in remaining cases
        *a = random_uni01() * 0.45 + 0.55;
    else //night is just dark, not very interesting
        *a = random_uni01() * 0.45;

    if(random_uni01() < 0.7) //70% sampling of "interesting" area
    {
        //non-uniform (truncated gaussian) sampling of beta
        //rejection sampling should only very rarely be useful; probability of beta outside [0; 1] very small
        *b = -1;
        while(*b < 0 || *b > 1)
        {
            *b = random_gauss(); //mean = 0; sigma = 1
            *b *= 0.025; //mean = 0; sigma = 1/40
            *b += 0.5; //mean = 0.5; sigma = 1/40
        }
    }
    else //30% sampling of the rest, still need to learn it
        *b = random_uni01();
}


FILE* fptr; //output file

void* thread(void* args)
{
    int n = *(int*)args;

    //seed random numbers separately for each thread
    srand(time(NULL) ^ getpid() ^ pthread_self());
    for(int i = 0; i < 42; i++) //otherwise similarity between first numbers
        rand();
    
    for(int i = 0; i < n; i++)
    {
        //sample non-uniformly
        double x, y, z, w;
        random_sample(&x, &y, &z, &w);

        //calculations
        double r, g, b;
        r = normalized_4param(COLORS.x, x, y, z, w);
        g = normalized_4param(COLORS.y, x, y, z, w);
        b = normalized_4param(COLORS.z, x, y, z, w);


        //lock for printing, so everything gets printed nicely
        pthread_mutex_lock(&print_mutex);

        //print h, alpha, beta, gamma (here x, y, z, w)
        fprintf(fptr, "%f, ", x);
        fprintf(fptr, "%f, ", y);
        fprintf(fptr, "%f, ", z);
        fprintf(fptr, "%f, ", w);

        //print results (red, green, blue)
        fprintf(fptr, "%f, ", r);
        fprintf(fptr, "%f, ", g);
        fprintf(fptr, "%f\n", b);

        pthread_mutex_unlock(&print_mutex);
    }

    return NULL;
}


#define NTHREADS 8

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        printf("Usage: sampler <NSAMPLES>\n");
        printf("NSAMPLES MUST be an integer, number of samples created\n");
        printf("Creates NSAMPLES samples in samples.txt\n");
        printf("Sample format: each sample is h, alpha, beta, gamma, red, green, blue\n");
        return 1;
    }

    //get number of samples
    int NSAMPLES = atoi(argv[1]);
    int NPERTHREAD = 1 + (NSAMPLES - 1) / NTHREADS; //round up

    //initialize print mutex and open file
    pthread_mutex_init(&print_mutex, NULL);
    fptr = fopen("samples.txt", "w");

    //launch threads
    pthread_t threads[NTHREADS];
    for(int i = 0; i < NTHREADS; i++)
        pthread_create(&(threads[i]), NULL, &thread, &NPERTHREAD);

    //wait for threads to finish
    for(int i = 0; i < NTHREADS; i++)
        pthread_join(threads[i], NULL);

    fclose(fptr);
}

//build command:
//gcc sampler.c -O3 -ffast-math -lm -s -DNDEBUG -o sampler.exe

