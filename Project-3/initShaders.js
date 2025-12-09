// initShaders.js
// Helper function for compiling and linking WebGL shaders

function initShaders(gl, vertexShaderId, fragmentShaderId) {
    const compileShader = (gl, shaderId, shaderType) => {
        const shaderScript = document.getElementById(shaderId);
        if (!shaderScript) {
            console.error(`Shader script with id '${shaderId}' not found`);
            return null;
        }

        const shader = gl.createShader(shaderType);
        gl.shaderSource(shader, shaderScript.text);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error(`Shader compilation error in ${shaderId}:`, gl.getShaderInfoLog(shader));
            return null;
        }

        return shader;
    };

    const vertexShader = compileShader(gl, vertexShaderId, gl.VERTEX_SHADER);
    const fragmentShader = compileShader(gl, fragmentShaderId, gl.FRAGMENT_SHADER);

    if (!vertexShader || !fragmentShader) {
        return null;
    }

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Shader program linking error:', gl.getProgramInfoLog(program));
        return null;
    }

    return program;
}
