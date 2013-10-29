#include "visualizer.hpp"
/* Visualizer Class
 * Holds the arrays containing vertices, indicies, normals and colors
 *
 */
#define XWIDTH 100.0f
#define ZWIDTH 100.0f
#define YWIDTH 100.0f

void display3DWrapper()
{
    Visualizer::getInstance().display3D();
}
void display2DWrapper()
{
    Visualizer::getInstance().display2D();
}
void displayCubeWrapper()
{
    Visualizer::getInstance().displayCube();
}
void ReshapeFunc3DWrapper(int width, int height)
{
    Visualizer::getInstance().Reshape3DFunc(width, height);
}
void ReshapeFunc2DWrapper(int width, int height)
{
    Visualizer::getInstance().Reshape2DFunc(width, height);
}
void keyHitWrapper(unsigned char key, int x, int y)
{
    Visualizer::getInstance().keyHit(key, x, y);
}
/*
 *  Singleton specific member functions
 */
Visualizer& Visualizer::getInstance()
{
    static Visualizer self;
    return self;
}
Visualizer::Visualizer():valid(false)
{

}

void Visualizer::setValues(bh_view* _bh, int w, int h, int d, int _cm, bool f, bool c, float _min, float _max)
{
    if (!valid){
        valid = true;
        pause = false;
        A = _bh;
        B = _bh->base;
        type = B->type;
        width = w;
        height = h;
        depth = d;
        cout << "w:" << width << ", h:" << height << ", depth: " << depth << endl;
        flat = f;
        cubes = c;
        min = _min;
        max = _max;
        nbPoints = width*height;
        nbQuads = (width-1)*(height-1);
        if (!cubes) {
            vertices = new float[3*nbPoints];
            normals = new float[3*nbPoints];
            indices = new int[nbQuads*4];
            colors = new float[nbPoints*3];
        }
        else
            vertices = new float[6*width*height*depth];

        cout << "A->ndim " << A->ndim << endl;
        cout << "A->shape " << A->shape[0] << ", " <<  A->shape[1] << ", " << A->shape[2] <<endl;
        cout << "A->stride " << A->stride[0] << ", " <<  A->stride[1] << ", " << A->stride[2] <<endl;
        cout << "A->start " << A->start << endl;
        cout << "Array:" << endl;
        for (int i = 0; i < A->shape[0]; i++){
            for (int j = 0; j < A->shape[1]; j++)
            {
                cout << ((float*)B->data)[A->start + i*A->stride[0]+j*A->stride[1]] << ", ";
            }
            cout << endl;
        }
        showLines = false;
        cm = colormaps[_cm];
        initOpenGL();
        if (!cubes){
            if (flat){
                computeVertices2D();

            }
            else{
                computeVertices3D();
            }
            computeIndices();
            updateColors();
            updateNormals();
        }
        else {
            computeVerticesCube();
        }
    }
    else {
        cout << "set Values should only be called once" << endl;
        exit(-1);
    }
}
Visualizer::~Visualizer()
{
  delete[] vertices;
  delete[] normals;
  delete[] indices;
  delete[] colors;
}
void Visualizer::run(bh_view* array)
{
    A = array;
    B = array->base;
    if (valid)
    {   if (!cubes)
        {
            if(flat)
                updateArray2D();
            else
                updateArray3D();
        }
        if (!pause)
        {
            glutMainLoopEvent();
            glutPostRedisplay();
        }
        else {
            glutMainLoop();
            initOpenGL();
        }
    }
    else
    {
        cout << "Visualizer not initialized!" << endl;
        exit(-1);
    }
}
void Visualizer::initOpenGL()
{

    r = 150.0f; /* radius for the camera */
    theta = M_PI/15; /* angle around 0y */
    phi =  M_PI/7; /* angle around 0z */

    /* camera position */
    if (flat){
        cameraPos.x = 0;
        cameraPos.y = r*sin(phi);
        cameraPos.z = 0;
    }
    else{
        cameraPos.x = r*cos(phi)*sin(theta);
        cameraPos.y = r*sin(phi);
        cameraPos.z = r*cos(phi)*cos(theta);
    }
    verticale.x = 0;
    verticale.y = 1;
    verticale.z = 0; 
    Vector3 cameraDir(3.5,0.0,3.5);
    char *myargv [1];
    int myargc=1;
    myargv [0]= strdup("visualizer");

    /* Initialize OpenGL */
    glutInit(&myargc, myargv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition (100, 100); 
    glutCreateWindow("Bohrium Visualizer");
    // Ensures that exiting the pause mode, glutLeaveMainLoop will not kill the application
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    /* Light */
    if (!flat){
        //glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
        //glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
        //glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
        //glLightfv(GL_LIGHT0, GL_POSITION, positionLight);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        //GLfloat mat_specular[] = { 1.0F,1.0F,1.0F,1.0F };
        //glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
        //glMateriali(GL_FRONT,GL_SHININESS,30);
    }


    /* Declaration of the callbacks */
    if (!cubes){
        if (flat){
            glutReshapeFunc(&ReshapeFunc2DWrapper);
            glutDisplayFunc(&display2DWrapper);

        }
        else{
            glutReshapeFunc(&ReshapeFunc3DWrapper);
            glutDisplayFunc(&display3DWrapper);
        }
    }
    else
    {
        glutReshapeFunc(&ReshapeFunc3DWrapper);
        glutDisplayFunc(&displayCubeWrapper);
    }
    glutKeyboardFunc(&keyHitWrapper);
    //glutKeyboardFunc(&keyHit);
    //glutMouseFunc(&mouseClicked);
    //glutMotionFunc(&mouseMoved);

}


void Visualizer::computeVertices2D()
{

  float dx = XWIDTH/ (float)(width-1);
  float dz = ZWIDTH/ (float)(height - 1);

  int i = 0;

  int j = 0;//width;

  for (int z = 0; z < height; z++)
  {
    for (int x = 0; x < width; x++)
    {
      vertices[3*i] = (float)(x*dx) - (XWIDTH/2.0f);
      vertices[3*i+1] = (float)(z*dz) - (ZWIDTH/2.0f);
      vertices[3*i+2] = 0.0f;
      j++;
      i++;
    }
  }
}
void Visualizer::computeVertices3D()
{

  float dx = XWIDTH/ (float)(width-1);
  float dz = ZWIDTH/ (float)(height - 1);

  int i = 0;

  int j = 0;//width;

  for (int z = 0; z < height; z++)
  {
    for (int x = 0; x < width; x++)
    {
      vertices[3*i] = (float)(x*dx) - (XWIDTH/2.0f);
      vertices[3*i+1] = ((float *)B->data)[j];
      vertices[3*i+2] = (float)(z*dz) - (ZWIDTH/2.0f);
      j++;
      i++;
    }
  }
}
void Visualizer::updateArray3D()
{
  //int j = 0;
  ////cout << "vertices[3*i+1] ";
  //for (int i = 0; i < nbPoints; i++)
  //{
  //  vertices[3*i+1] = ((float*)B->data)[j];
  //  //cout << 3*i+1 << " ";

  //}
  //cout << endl;
  //cout << "[i*A->shape[0] + j] ";
  for (int i = 0; i < A->shape[0]; i++){
      for (int j = 0; j < A->shape[1]; j++)
      {
  //        cout << ((float*)B->data)[A->start + i*A->stride[0]+j*A->stride[1]] << ", ";
          vertices[3*(i * width + j)+1] = ((float*)B->data)[A->start + i*A->stride[0]+j*A->stride[1]];
  //          cout << (3*(i * width + j)+1) << ", ";
      }
  }
  //cout << endl;

  updateColors();
  updateNormals();
}
void Visualizer::updateArray2D()
{
  updateColors();
}
void Visualizer::computeIndices()
{
  int k = 0;
  for (int i = 0; i < nbQuads; i++)
  {
    if (i > 0 && i % (width-1) == 0)
      k++;
    indices[4*i] = k;
    indices[4*i+1] = k+1;
    indices[4*i+2] = width+1+k;
    indices[4*i+3] = width+k;

    k++;
  }
}

void Visualizer::updateNormals()
{
  Vector3 v1;
  Vector3 v2;
  Vector3 v3;
  for (int i = 0; i < nbQuads ; i++)
  {
    v1.x = vertices[3*i+3] - vertices[3*i];
    v1.y = vertices[3*i+4] - vertices[3*i+1];
    v1.z = vertices[3*i+5] - vertices[3*i+2];
    v2.x = vertices[3*i+width*3] - vertices[3*i];
    v2.y = vertices[3*i+width*3+1] - vertices[3*i+1];
    v2.z = vertices[3*i+width*3+2] - vertices[3*i+2];
    v3 = v2.crossProduct(v1);
    v3.normalize();
    /* compute only the left of the quad */
    normals[3*i] = v3.x;
    normals[3*i+1] = v3.y;
    normals[3*i+2] = v3.z;
    normals[3*i+width*3] = v3.x;
    normals[3*i+width*3+1] = v3.y;
    normals[3*i+width*3+2] = v3.z;
  }

  int i = nbQuads;
  normals[3*i+3] = normals[3*i];
  normals[3*i+4] = normals[3*i+1];
  normals[3*i+5] = normals[3*i+2];

  normals[3*i+width*3+3] = normals[3*i+width*3];
  normals[3*i+width*3+4] = normals[3*i+width*3+1];
  normals[3*i+width*3+5] = normals[3*i+width*3+2];
}

void Visualizer:: updateColors()
{
  int k = 0;
  for (int i = 0; i < A->shape[0]; i++){
      for (int j = 0; j < A->shape[1]; j++)
      {
          float v = ((float*)B->data)[A->start + i*A->stride[0]+j*A->stride[1]];
          v =  ((v- min) / (max - min));
          colors[k] = interpolateColor(v, cm.red);
          colors[k+1] = interpolateColor(v, cm.green);
          colors[k+2] = interpolateColor(v, cm.blue);
        k += 3;
      }
  }

}


float Visualizer::interpolateColor(float v, const float (* rgb)[3])
{
    if (v <= 0.0)
    {
        return rgb[0][2];
    }
    else if (v >= 1.0){
        return rgb[cm.size-1][1];
    }
    else {
        int low = 0;
        int high = cm.size - 1;
        while (low <= high) 
        {
            // invariants: value > A[i] for all i < low
            //             value <= A[i] for all i > high
            int mid = (low + high) / 2;
            if (rgb[mid][0] >= v)
                high = mid - 1;
            else
                low = mid + 1;
        }
        int res = low - 1;
        float x_0 = rgb[res][0];
        float x_1 = rgb[res+1][0];
        float y_0 = rgb[res][2];
        float y_1 = rgb[res+1][1];
        return  y_0 + (y_1 - y_0) * ((v-x_0)/(x_1- x_0));
    }
    return 0.0f;
}
void Visualizer::computeVerticesCube()
{
    uint64_t i = 0;
    for (int xi = 0; xi < width; xi++ )
    {
        for (int yi = 0; yi < height; yi++ )
        {
            for (int zi = 0; zi < depth; zi++ )
            {
                uint64_t offset = (xi * width + yi) * depth + zi;
                offset = i;
                float dx = (XWIDTH/2.0f)/ (float)(width);
                float dy = (YWIDTH/2.0f)/ (float)(height);
                float dz = dx;//(ZWIDTH/2.0f)/ (float)(depth);
                int x = XWIDTH/4.0f - (dx * xi);int y = YWIDTH/4.0f - (dy * yi);int z =  ZWIDTH/4.0f - (dz *zi);

                vertices[offset] = x + dx/2.0f;
                vertices[offset + 1] = y + dy/2.0f;
                vertices[offset + 2] = z + dz/2.0f;
                vertices[offset + 3] = x - dx/2.0f;
                vertices[offset + 4] = y - dy/2.0f;
                vertices[offset + 5] = z - dz/2.0f;

                i +=6;
            }

        }

    }
}
void Visualizer::drawCube(int x, int y, int z, float v)
{
    // x, y, z is the 3 dimensional points defining the center of the Cube.
    //dx, dy, dz, is the spacing between each entry in the matrix
    /*
    y
    ^         p_1    _________________________  p_2
    |               / _____________________  /|
    |              / / ___________________/ / |
    |             / / /| |               / /  |
    |            / / / | |              / / . |
    |           / / /| | |             / / /| |
    |          / / / | | |            / / / | |
    |         / / /  | | |      p_4  / / /| | |
    |        / /_/__________________/ / / | | |
    | p_3   /________________________/ /  | | |
    |       | ______________________ | |  | | |
    0.0     | | |    | | |_________| | |__| | |
    |       | | |p_5 | |___________| | |____| | p_6
    |       | | |   / / ___________| | |_  / /
    |       | | |  / / /           | | |/ / /
    |       | | | / / /            | | | / /
    |       | | |/ / /             | | |/ /
    |       | | | / /              | | ' /
    |       | | |/_/_______________| |  /
    |       | |____________________| | /
    |   p_8 |________________________|/ p_7
    |
    -------------------0.0-----------------> x
    */
    dx = (XWIDTH/2.0f)/ (float)(width);
    dy = (YWIDTH/2.0f)/ (float)(height);
    dz = dy;//(ZWIDTH/2.0f)/ (float)(depth);
    v =((v - min) / (max - min));
    float xi = (XWIDTH/4.0f) - (dx * x); float yi = (YWIDTH/4.0f) - (dy * y); float zi =  (ZWIDTH/4.0f) - (dz *z);
    Vector3 p1, p2, p3, p4;
    p1.x = xi - (dx/2.0f); p1.y = yi + (dy/2.0f);p1.z = zi - (dz/2.0f);
    p2.x = xi + (dx/2.0f); p2.y = yi + (dy/2.0f);p2.z = zi - (dz/2.0f);
    p3.x = xi - (dx/2.0f); p3.y = yi + (dy/2.0f);p3.z = zi + (dz/2.0f);
    p4.x = xi + (dx/2.0f); p4.y = yi + (dy/2.0f);p4.z = zi + (dz/2.0f);
    float negy = yi - (dy/2.0f);

    glBegin(GL_QUADS);
    // Top
    glColor3f(interpolateColor(v, cm.red), interpolateColor(v, cm.green), interpolateColor(v, cm.blue));
    glNormal3f(0.0f, 1.0f, 0.0f);
    glVertex3f(p4.x, p4.y, p4.z); //p4
    glVertex3f(p2.x, p2.y, p2.z); //p2
    glVertex3f(p1.x, p1.y, p1.z); //p1
    glVertex3f(p3.x, p3.y, p3.z); //p3
    // Right
    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(p2.x, negy, p2.z); // p6
    glVertex3f(p2.x, p2.y, p2.z); // p2
    glVertex3f(p4.x, p4.y, p4.z); // p4
    glVertex3f(p4.x, negy, p4.z); // p7
    // Bottom
    glNormal3f(0.0f, -1.0f, 0.0f);
    glVertex3f(p2.x, negy, p2.z); // p6
    glVertex3f(p4.x, negy, p4.z); // p7
    glVertex3f(p3.x, negy, p3.z); // p8
    glVertex3f(p1.x, negy, p1.z); // p5
    // Left
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(p3.x, negy, p3.z); // p8
    glVertex3f(p3.x, p3.y, p3.z); // p3
    glVertex3f(p1.x, p1.y, p1.z); // p1
    glVertex3f(p1.x, negy, p1.z); // p5
    // Back
    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(p2.x, negy, p2.z); // p6
    glVertex3f(p2.x, p2.y, p2.z); // p2
    glVertex3f(p1.x, p1.y, p1.z); // p1
    glVertex3f(p1.x, negy, p1.z); // p5
    // Front
    glNormal3f(0.0f, 0.0f, 1.0f);
    glVertex3f(p4.x, negy, p4.z); // p7
    glVertex3f(p4.x, p4.y, p4.z); // p4
    glVertex3f(p3.x, p3.y, p3.z); // p3
    glVertex3f(p3.x, negy, p3.z); // p8
    glEnd();
}
void Visualizer::displayCube(){
      //  Clear screen and Z-buffer
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    if (showLines)
      glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
    else
      glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);


    glLoadIdentity();
    gluLookAt(cameraPos.x, cameraPos.y,cameraPos.z, /* only the camera move */
      cameraDir.x, cameraDir.y, cameraDir.z,
      verticale.x,verticale.y,verticale.z);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int z = 0; z < depth; z++)
            {
                if (((float *)B->data)[(x * width + y) * depth + z] > 0.0)
                {
                    drawCube(x, y , z, ((float *)B->data)[(x * width + y) * depth + z]);
                }
            }
        }
    }

    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}

void Visualizer::display2D()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();


  if (showLines)
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
  else
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, vertices);
  glNormalPointer(GL_FLOAT, 0, normals);
  glColorPointer(3, GL_FLOAT, 0, colors);
  glDrawElements(GL_QUADS, nbQuads*4, GL_UNSIGNED_INT, indices);
  glFlush();


  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
  glutSwapBuffers();

  /* Update again and again */
  glutPostRedisplay();
}
void Visualizer::display3D()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();

  gluLookAt(cameraPos.x, cameraPos.y,cameraPos.z, /* only the camera move */
      cameraDir.x, cameraDir.y, cameraDir.z,
      verticale.x,verticale.y,verticale.z);


  if (showLines)
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
  else
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, vertices);
  glNormalPointer(GL_FLOAT, 0, normals);
  glColorPointer(3, GL_FLOAT, 0, colors);
  glDrawElements(GL_QUADS, nbQuads*4, GL_UNSIGNED_INT, indices);
  glFlush();


  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
  //glFinish();
  glutSwapBuffers();

  /* Update again and again */
  glutPostRedisplay();
}
void Visualizer::updateCamera()
{
  cameraPos.x = r*cos(phi)*sin(theta);
  cameraPos.y = r*sin(phi);
  cameraPos.z = r*cos(phi)*cos(theta);
}
// When a key is hit
void Visualizer::keyHit(unsigned char key, int x, int y)
{

  Vector3 vs(-1,0,0);
  switch (key)
  {
    case 'l' :
      showLines = !showLines;
      break;
    case 'q' :
      exit(0);
      break;
    case 'd':
      theta +=  -0.1f;
      updateCamera();
      break;
    case 'a':
      theta +=  0.1f;
      updateCamera();
      break;
    case 'w':
      phi += - 0.1f;
      updateCamera();
      verticale = vs.crossProduct(cameraPos);
      verticale.normalize();
      break;
    case 's':
      phi +=  0.1f;
      updateCamera();
      verticale = vs.crossProduct(cameraPos);
      verticale.normalize();
      break;
    case 'x':
      r += 5.0f;
      updateCamera();
      break;
    case 'z':
      r += -5.0f;
      updateCamera();
      break;
    case 'p':
      if (pause)
      {
          pause = false;
          glutLeaveMainLoop();
      }
      else
      {
          pause = true;
      }
      break;

  }
}


void Visualizer::Reshape3DFunc(int width, int height)
{
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    gluPerspective(40, width / (float) height, 1, 280);
    glViewport(0, 0, width, height);

    glMatrixMode(GL_MODELVIEW);
    glutPostRedisplay();
}
void Visualizer::Reshape2DFunc(int w, int h)
{
    glViewport(0,0,(GLsizei) w, (GLsizei) h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-100.0, 100.0, -100.0, 100.0, -20.0, 20.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}
