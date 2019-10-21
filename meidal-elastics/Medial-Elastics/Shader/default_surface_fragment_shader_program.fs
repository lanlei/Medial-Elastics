#ifdef GL_ES
// set default precision to medium
precision mediump int;
precision mediump float;
#endif

uniform sampler2D texture_sampler;

struct GeneralLight
{
   vec3 position;
   vec3 ambient_color;
   vec3 diffuse_color;
   vec3 specular_color;
};

uniform GeneralLight light0;


uniform vec3 viewPos;
uniform int use_texture;


uniform vec3 surfaceColor;

varying vec3 Frag_pos;
varying vec2 texcoord;


void main()
{
	if(use_texture == 1)
		surfaceColor = texture(texture_sampler, texcoord).rgb;
		
	vec3 normal = normalize(cross(dFdx(Frag_pos), dFdy(Frag_pos)));
	vec3 ambient = 0.6 * light0.ambient_color;
	vec3 lightDir = normalize(light0.position - Frag_pos);
	float diff = max(dot(normal, lightDir), 0.0);
	vec3 diffuse = 1.0*diff * light0.diffuse_color;
	vec3 viewDir = normalize(viewPos - Frag_pos);
	vec3 halfwayDir = normalize(lightDir + viewDir); 
	float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
	vec3 specular = 0.3 * spec * light0.specular_color;

	vec3 result = (ambient + diffuse + specular)* surfaceColor;	
	gl_FragColor = vec4(result , 1.0);	
}
