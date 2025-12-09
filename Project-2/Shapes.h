/////////////////////////////////////////////////////////////////////////////
//
//  --- Shapes.h ---
//
//  A collection of helper classes for doing SDF (signed-distance function)
//    computations
//

#pragma once

#ifndef __SHAPES_H__
#define __SHAPES_H__

#include <cmath>
//#include <format>
#include <iostream>

//---------------------------------------------------------------------------
//
//  vec3 - a three-component vector, used for storing 3D points and vectors
//

struct vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    vec3() = default;

    vec3(float v) : x(v), y(v), z(v)
        { /* Empty */}

    vec3(float x, float y, float z) : x(x), y(y), z(z)
        { /* Empty */}

    vec3& operator -= (const vec3& v)
        { x -= v.x;  y -= v.y;  z -= v.z; return *this; }

    vec3& operator *= (const vec3& v)
        { x *= v.x;  y *= v.y;  z *= v.z; return *this; }
    
    float length() const
        { return std::sqrt(x*x + y*y + z*z); }

    friend std::ostream& operator << (std::ostream& os, const vec3& v)
        { return os << "( " << v.x << ", " << v.y << ", " << v.z << " )"; }
};

//---------------------------------------------------------------------------
//
//  Sphere - a class for positioning a sphere
//

struct Sphere {
    vec3  center;
    float radius;

    Sphere(const float radius = 1.0f) : radius(radius)
        { /* Empty */}

    Sphere(const vec3& p, float radius = 1.0) : center(p), radius(radius)
        { /* Empty */}

    friend std::ostream& operator << (std::ostream& os, const Sphere& s) {
        return os << "center: " << s.center << "  radius: " << s.radius;
    }
};

#endif // __SHAPES_H__