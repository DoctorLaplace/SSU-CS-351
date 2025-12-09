# Project 3: WebGL Shaders

This project demonstrates procedural shape generation using WebGL vertex and fragment shaders. Each version builds progressively on the previous one.

## Versions

### [Version 1: Wireframe Triangle](v1-triangle.html)

A simple equilateral triangle created procedurally using trigonometry in the vertex shader. Rendered as a yellow line loop.

**Key Concepts:**
- Basic vertex shader with procedural geometry
- Computing vertex positions using `cos()` and `sin()`
- Fragment shader with flat color

![Wireframe Triangle](r1.png)

---

### [Version 2: 10-Sided Filled Polygon](v2-polygon.html)

A filled 10-sided polygon created using a triangle fan primitive.

**Key Concepts:**
- Triangle fan rendering (`gl.TRIANGLE_FAN`)
- Procedural vertex generation for N-sided shapes
- Angle calculation for even distribution

![10-Sided Polygon](r2.png)

---

### [Version 3: Five-Pointed Star](v3-star.html)

A classic five-pointed star created by modulating vertex radius based on even/odd vertex IDs.

**Key Concepts:**
- Radius modulation based on vertex ID
- Using `mod()` function for even/odd detection
- Odd vertices at full radius (outer points)
- Even vertices at reduced radius (inner points, 0.4)
- Explicit handling of center vertex

![Five-Pointed Star](r3.png)

---

### [Version 4: Rotating Star](v4-rotating-star.html)

The five-pointed star now rotates continuously using animation.

**Key Concepts:**
- Time-based animation using `requestAnimationFrame()`
- Time uniform variable (`t`) passed to shader
- Incorporating time into angle computation for rotation

![Rotating Star](r4.png)

---

### [Version 5: Colored Rotating Star (Extra Credit)](v5-colored-star.html)

The rotating star with beautiful color interpolation from gold center to deep red edges.

**Key Concepts:**
- Shader variable export/import (`out` from vertex, `in` to fragment)
- Color interpolation using `mix()` function
- Radius-based color gradient

![Colored Rotating Star](r5.png)

---

## Technical Implementation

### Shader Pipeline
- **Vertex Shader**: Generates vertex positions procedurally using trigonometry and time-based transformations
- **Fragment Shader**: Handles coloring, from simple flat colors to interpolated gradients

### Progressive Development
Each version adds new concepts while maintaining the foundation:
1. Basic procedural geometry → 2. Filled shapes → 3. Conditional logic → 4. Animation → 5. Inter-shader communication

## Files
- `initShaders.js` - Helper function for compiling and linking shaders
- `v1-triangle.html` - Wireframe triangle
- `v2-polygon.html` - 10-sided polygon
- `v3-star.html` - Five-pointed star
- `v4-rotating-star.html` - Animated rotating star
- `v5-colored-star.html` - Colored animated star (extra credit)

## Running the Applications
Simply open any HTML file in a WebGL 2.0 capable browser. No web server required for these simple applications.
