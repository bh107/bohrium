#ifndef __PROTOTYPE
#define __PROTOTYPE

#include <iostream>
#include <stdlib.h>
#include <GL/freeglut.h>
#include <iostream>
#include <sys/resource.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <bh.h>
#include "Vector3.hpp"
#include "colormaps.hpp"
#define max(a,b) (a>=b?a:b)
#define min(a,b) (a<=b?a:b)

using namespace std;


class Visualizer
{
  private:
    int width, height, depth, nbPoints;
    bool showLines;
    bool flat;
    bool cubes;
    bool pause;
    float r, theta, phi;
    float dx;
    float dy;
    float dz;
    Visualizer();
    bool valid;


  public:

    ~Visualizer();
    static Visualizer& getInstance();
    void run(bh_view* array);
    void setValues(bh_view* array, int width, int height, int depth, int cm, bool flat, bool cubes, float min, float max);

    // OpenGL Methods and variables
    void initOpenGL();
    void updateNormals();
    void updateColors();
    float interpolateColor(float value, const float (* rgb)[3]);

    void updateArray3D();
    void updateArray2D();
    void computeIndices();
    void computeVertices3D();
    void computeVertices2D();
    void computeVerticesCube();
    void display3D(void);
    void display2D(void);
    void displayCube(void);
    void drawCube(int x, int y, int z, float v);
    void Reshape3DFunc(int width, int height);
    void Reshape2DFunc(int width, int height);
    void keyHit(unsigned char key, int x, int y);
    void updateCamera();
    Vector3 cameraPos;
    Vector3 cameraDir;
    Vector3 verticale;

    bh_view* A;
    bh_base* B;
    bh_type type;
    float* vertices;
    float* normals;
    int* indices;
    float* colors;

    CMStruct cm;
    int nbQuads;
    float min;
    float max;
};

#endif
