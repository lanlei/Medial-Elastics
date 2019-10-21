#include <iostream>
#include "ScreenFontType.h"

namespace ScreenTextPainter
{

	TextPainter::TextPainter(): indexBuf(QOpenGLBuffer::IndexBuffer)
	{
		index.resize(6);
		index[0] = 0;
		index[1] = 1;
		index[2] = 2;
		index[3] = 0;
		index[4] = 2;
		index[5] = 3;
	}

	TextPainter::~TextPainter()
	{
		arrayBuf.destroy();
		indexBuf.destroy();

		for (int i = 0; i < textures.size(); i++)
			delete textures[i];
		textures.clear();
	}

	void TextPainter::generateFont(string filename, float height)
	{
		initializeOpenGLFunctions();
		arrayBuf.create();
		indexBuf.create();
		if(FT_Init_FreeType(&ft))
			cout << "ERROR::FREETYPE: Could not init FreeType Library" << endl;
		if(FT_New_Face(ft, filename.c_str(), 0, &face))
			cout << "ERROR::FREETYPE: Failed to load font" << endl;
		FT_Set_Pixel_Sizes(face, 0, height);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		for (GLubyte c = 0; c < 128; c++)
		{
			// load glyph
			if (FT_Load_Char(face, c, FT_LOAD_RENDER))
			{
				std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
				continue;
			}
			// Generate texture
			GLuint texture;
			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexImage2D(
				GL_TEXTURE_2D,
				0,
				GL_RED,
				face->glyph->bitmap.width,
				face->glyph->bitmap.rows,
				0,
				GL_RED,
				GL_UNSIGNED_BYTE,
				face->glyph->bitmap.buffer
			);
			// Set texture options
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			Character character = {
				texture,
				Vector2i(face->glyph->bitmap.width, face->glyph->bitmap.rows),
				Vector2i(face->glyph->bitmap_left, face->glyph->bitmap_top),
				face->glyph->advance.x
			};
			charList.insert(std::pair<GLchar, Character>(c, character));
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		FT_Done_Face(face);
		FT_Done_FreeType(ft);

		indexBuf.bind();
		indexBuf.allocate(index.data(), 6 * sizeof(GLushort));
	}

	void TextPainter::renderText(QOpenGLShaderProgram * program, string text, GLfloat x, GLfloat y, GLfloat scale, QVector3D color)
	{
		glEnable(GL_BLEND);
	   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glActiveTexture(GL_TEXTURE0);
		program->setUniformValue("textColor", color);
		std::string::const_iterator c;
		for (c = text.begin(); c != text.end(); c++)
		{
			Character ch = charList[*c];
			GLfloat xpos = x + ch.bearing[0] * scale;
			GLfloat ypos = y - (ch.size[1] - ch.bearing[1]) * scale;
			GLfloat w = ch.size[0] * scale;
			GLfloat h = ch.size[1] * scale;

			fontData.resize(4);
			fontData[0].pos[0] = xpos;
			fontData[0].pos[1] = ypos + h;
			fontData[0].tex[0] = 0.0;
			fontData[0].tex[1] = 0.0;

			fontData[1].pos[0] = xpos;
			fontData[1].pos[1] = ypos;
			fontData[1].tex[0] = 0.0;
			fontData[1].tex[1] = 1.0;

			fontData[2].pos[0] = xpos + w;
			fontData[2].pos[1] = ypos;
			fontData[2].tex[0] = 1.0;
			fontData[2].tex[1] = 1.0;

			fontData[3].pos[0] = xpos + w;
			fontData[3].pos[1] = ypos + h;
			fontData[3].tex[0] = 1.0;
			fontData[3].tex[1] = 0.0;

			glBindTexture(GL_TEXTURE_2D, ch.textureId);
			arrayBuf.bind();
			arrayBuf.allocate(fontData.data(), fontData.size() * sizeof(FontCoords));
			// Offset for position
			quintptr offset = 0;
			int vertexLocation = program->attributeLocation("a_position");
			program->enableAttributeArray(vertexLocation);
			program->setAttributeBuffer(vertexLocation, GL_FLOAT, offset, 2, sizeof(FontCoords));
			offset += sizeof(QVector2D);
			int texcoordLocation = program->attributeLocation("a_texcoord");
			program->enableAttributeArray(texcoordLocation);
			program->setAttributeBuffer(texcoordLocation, GL_FLOAT, offset, 2, sizeof(FontCoords));
			indexBuf.bind();
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
			x += (ch.advance >> 6) * scale;
		}
		glDisable(GL_BLEND);
	}
}


