#ifdef GL_ES
// set default precision to medium
precision mediump int;
precision mediump float;
#endif

uniform mat4 project_matrix;
uniform mat4 view_matrix;

attribute vec3 a_position;
attribute vec2 a_texcoord;

varying vec3 Frag_pos;
varying vec2 texcoord;


void main()
{
	gl_Position = project_matrix * view_matrix * vec4(a_position, 1.0);
	Frag_pos = 	a_position;
	texcoord = a_texcoord;
}