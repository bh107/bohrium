#ifndef VECTOR_3_H
#define VECTOR_3_H

class Vector3
{
  public:
    float x;
    float y;
    float z;

    Vector3();
    Vector3(float x,float y,float z);
    Vector3(const Vector3 & v);

    Vector3& operator= (const Vector3 & v);

    Vector3& operator+= (const Vector3 & v);
    Vector3&  operator-= (const Vector3 & v);
    Vector3&  operator*= (const float a);
    Vector3&  operator/= (const float a);

    Vector3 operator+ (const Vector3 & v);
    Vector3 operator- (const Vector3 & v);
    Vector3 operator* (const float a);
    Vector3 operator/ (const float a);
    float& operator()(int i);


    Vector3 crossProduct(const Vector3 & v);
    void normalize();
    float norm();
};

#endif
