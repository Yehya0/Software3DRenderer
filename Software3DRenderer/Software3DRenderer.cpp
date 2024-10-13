#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <limits>
#include <chrono>
#include <thread>
#include <random>
#include <cstring>
#include <sstream>
#include <iostream>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
void enableANSIColors() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;
    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) return;
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
}
#else
void enableANSIColors() {}
#endif

template <typename T>
T clamp(T value, T minValue, T maxValue) {
    return std::min(std::max(value, minValue), maxValue);
}

class Vec3 {
public:
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }

    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }

    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    float length() const { return std::sqrt(x * x + y * y + z * z); }

    Vec3 normalized() const {
        float l = length();
        return l > 0 ? *this / l : *this;
    }

    Vec3 rotate(float angle, const Vec3& axis) const {
        float c = cos(angle);
        float s = sin(angle);
        float t = 1 - c;
        Vec3 a = axis.normalized();
        return Vec3(
            (t * a.x * a.x + c) * x + (t * a.x * a.y - s * a.z) * y + (t * a.x * a.z + s * a.y) * z,
            (t * a.x * a.y + s * a.z) * x + (t * a.y * a.y + c) * y + (t * a.y * a.z - s * a.x) * z,
            (t * a.x * a.z - s * a.y) * x + (t * a.y * a.z + s * a.x) * y + (t * a.z * a.z + c) * z
        );
    }

    friend Vec3 operator*(float s, const Vec3& v) { return v * s; }
};

class Mat4 {
public:
    std::array<float, 16> elements;

    Mat4() { elements.fill(0); }

    static Mat4 identity() {
        Mat4 mat;
        mat.elements[0] = mat.elements[5] = mat.elements[10] = mat.elements[15] = 1.0f;
        return mat;
    }

    static Mat4 perspective(float fov, float aspect, float nearPlane, float farPlane) {
        Mat4 mat;
        float tanHalfFov = tan(fov / 2);
        mat.elements[0] = 1 / (aspect * tanHalfFov);
        mat.elements[5] = 1 / tanHalfFov;
        mat.elements[10] = -(farPlane + nearPlane) / (farPlane - nearPlane);
        mat.elements[11] = -1.0f;
        mat.elements[14] = -(2 * farPlane * nearPlane) / (farPlane - nearPlane);
        return mat;
    }

    static Mat4 rotation(float angle, const Vec3& axis) {
        Mat4 mat = Mat4::identity();
        float c = cos(angle);
        float s = sin(angle);
        float t = 1 - c;
        Vec3 a = axis.normalized();

        mat.elements[0] = c + a.x * a.x * t;
        mat.elements[1] = a.x * a.y * t + a.z * s;
        mat.elements[2] = a.x * a.z * t - a.y * s;

        mat.elements[4] = a.x * a.y * t - a.z * s;
        mat.elements[5] = c + a.y * a.y * t;
        mat.elements[6] = a.y * a.z * t + a.x * s;

        mat.elements[8] = a.x * a.z * t + a.y * s;
        mat.elements[9] = a.y * a.z * t - a.x * s;
        mat.elements[10] = c + a.z * a.z * t;

        return mat;
    }

    Vec3 transform(const Vec3& v) const {
        float x = v.x * elements[0] + v.y * elements[4] + v.z * elements[8] + elements[12];
        float y = v.x * elements[1] + v.y * elements[5] + v.z * elements[9] + elements[13];
        float z = v.x * elements[2] + v.y * elements[6] + v.z * elements[10] + elements[14];
        float w = v.x * elements[3] + v.y * elements[7] + v.z * elements[11] + elements[15];

        if (w != 0.0f) {
            x /= w;
            y /= w;
            z /= w;
        }

        return Vec3(x, y, z);
    }

    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        result.elements.fill(0);
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                for (int i = 0; i < 4; ++i)
                    result.elements[row + col * 4] += elements[row + i * 4] * other.elements[i + col * 4];
        return result;
    }
};

struct Triangle {
    std::array<Vec3, 3> vertices;
    Vec3 color;
    Vec3 normal;
    float specular;
    float reflectivity;
};

struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec3 color;
    float life;
    float size;
};

struct Light {
    Vec3 position;
    Vec3 color;
    float intensity;
};

class Renderer {
private:
    int width, height;
    std::vector<Vec3> frameBuffer;
    std::vector<float> depthBuffer;
    Mat4 projectionMatrix;
    std::vector<Light> lights;
    std::vector<Particle> particles;
    std::mt19937 rng;
    std::vector<std::string> asciiChars;

public:
    Renderer(int w, int h) : width(w), height(h), frameBuffer(w* h), depthBuffer(w* h),
        rng(std::random_device{}()) {
        projectionMatrix = Mat4::perspective(M_PI / 3.0f, static_cast<float>(w) / h, 0.1f, 100.0f);
        asciiChars = { " ", ".", ":", "-", "=", "+", "*", "#", "%", "@" };

        lights.push_back({ Vec3(5, 5, -10), Vec3(1, 1, 1), 1.0f });
        lights.push_back({ Vec3(-5, -5, -10), Vec3(0.5f, 0.5f, 1), 0.7f });
    }

    void clear() {
        std::fill(frameBuffer.begin(), frameBuffer.end(), Vec3(0, 0, 0));
        std::fill(depthBuffer.begin(), depthBuffer.end(), std::numeric_limits<float>::infinity());
    }

    Vec3 barycentric_coordinates(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& p) {
        Vec3 v0 = b - a, v1 = c - a, v2 = p - a;
        float d00 = v0.dot(v0);
        float d01 = v0.dot(v1);
        float d11 = v1.dot(v1);
        float d20 = v2.dot(v0);
        float d21 = v2.dot(v1);
        float denom = d00 * d11 - d01 * d01;

        if (denom == 0.0f) return Vec3(-1, -1, -1);

        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;
        return Vec3(u, v, w);
    }

    void drawTriangle(const Triangle& triangle, const Mat4& modelViewMatrix) {
        std::array<Vec3, 3> projectedVertices;
        Vec3 faceNormal = modelViewMatrix.transform(triangle.normal).normalized();

        for (int i = 0; i < 3; ++i) {
            Vec3 v = modelViewMatrix.transform(triangle.vertices[i]);
            v = projectionMatrix.transform(v);
            v.x = (v.x + 1.0f) * 0.5f * width;
            v.y = (1.0f - (v.y + 1.0f) * 0.5f) * height;
            projectedVertices[i] = v;
        }

        Vec3 v0 = projectedVertices[0], v1 = projectedVertices[1], v2 = projectedVertices[2];

        if (((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)) < 0)
            return;

        int minX = std::max(0, static_cast<int>(std::min({ v0.x, v1.x, v2.x })));
        int minY = std::max(0, static_cast<int>(std::min({ v0.y, v1.y, v2.y })));
        int maxX = std::min(width - 1, static_cast<int>(std::max({ v0.x, v1.x, v2.x })));
        int maxY = std::min(height - 1, static_cast<int>(std::max({ v0.y, v1.y, v2.y })));

        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                Vec3 p(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, 0.0f);
                Vec3 barycentric = barycentric_coordinates(v0, v1, v2, p);
                if (barycentric.x >= 0 && barycentric.y >= 0 && barycentric.z >= 0) {
                    float depth = v0.z * barycentric.x + v1.z * barycentric.y + v2.z * barycentric.z;
                    int idx = y * width + x;
                    if (depth < depthBuffer[idx]) {
                        depthBuffer[idx] = depth;

                        Vec3 worldPos = triangle.vertices[0] * barycentric.x +
                            triangle.vertices[1] * barycentric.y +
                            triangle.vertices[2] * barycentric.z;

                        Vec3 viewDir = (Vec3(0, 0, -1) - worldPos).normalized();

                        Vec3 color = Vec3(0, 0, 0);
                        for (const auto& light : lights) {
                            Vec3 lightDir = (light.position - worldPos).normalized();
                            float diffuse = std::max(0.0f, faceNormal.dot(lightDir));

                            Vec3 reflectDir = (faceNormal * 2.0f * faceNormal.dot(lightDir) - lightDir).normalized();
                            float specular = std::pow(std::max(0.0f, viewDir.dot(reflectDir)), triangle.specular);

                            color = color + light.color * light.intensity * (triangle.color * diffuse + Vec3(1, 1, 1) * specular * triangle.reflectivity);
                        }

                        frameBuffer[idx] = color;
                    }
                }
            }
        }
    }

    void updateParticles(float deltaTime) {
        for (auto& particle : particles) {
            particle.position = particle.position + particle.velocity * deltaTime;
            particle.life -= deltaTime;
            particle.color.y = std::max(0.0f, particle.color.y - 0.01f);
            particle.size = std::max(0.1f, particle.size - 0.01f);
        }

        particles.erase(
            std::remove_if(particles.begin(), particles.end(),
                [](const Particle& p) { return p.life <= 0; }),
            particles.end()
        );

        if (particles.size() < 2000) {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            particles.push_back({
                Vec3(dist(rng), dist(rng), dist(rng)) * 3.0f,
                Vec3(dist(rng), std::abs(dist(rng)), dist(rng)),
                Vec3(1.0f, 0.7f + dist(rng) * 0.3f, 0.0f),
                1.0f + dist(rng),
                0.5f + std::abs(dist(rng)) * 0.5f
                });
        }
    }

    void drawParticles(const Mat4& viewMatrix) {
        for (const auto& particle : particles) {
            Vec3 viewPos = viewMatrix.transform(particle.position);
            Vec3 projectedPos = projectionMatrix.transform(viewPos);

            if (projectedPos.z < -1 || projectedPos.z > 1) continue;

            int x = static_cast<int>((projectedPos.x + 1.0f) * 0.5f * width);
            int y = static_cast<int>((1.0f - (projectedPos.y + 1.0f) * 0.5f) * height);

            if (x >= 0 && x < width && y >= 0 && y < height) {
                int size = static_cast<int>(particle.size * 3);
                for (int dy = -size; dy <= size; ++dy) {
                    for (int dx = -size; dx <= size; ++dx) {
                        int px = x + dx;
                        int py = y + dy;
                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            int idx = py * width + px;
                            float dist = std::sqrt(dx * dx + dy * dy) / size;
                            if (dist <= 1) {
                                float alpha = 1 - dist;
                                frameBuffer[idx] = frameBuffer[idx] * (1 - alpha) + particle.color * alpha;
                            }
                        }
                    }
                }
            }
        }
    }

    void render() const {
        std::cout << "\x1b[H";
        std::ostringstream output;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Vec3 color = frameBuffer[y * width + x];
                float intensity = (color.x + color.y + color.z) / 3.0f;
                int index = static_cast<int>(intensity * (asciiChars.size() - 1));
                index = clamp(index, 0, static_cast<int>(asciiChars.size()) - 1);

                int r = static_cast<int>(std::min(color.x * 255.0f, 255.0f));
                int g = static_cast<int>(std::min(color.y * 255.0f, 255.0f));
                int b = static_cast<int>(std::min(color.z * 255.0f, 255.0f));

                output << "\x1b[38;2;" << r << ";" << g << ";" << b << "m" << asciiChars[index] << "\x1b[0m";
            }
            output << "\n";
        }
        std::cout << output.str();
        std::cout.flush();
    }
};

std::vector<Triangle> createScene() {
    std::vector<Triangle> scene;

    auto addCube = [&scene](const Vec3& position, float size, const Vec3& color, float specular, float reflectivity) {
        std::array<Vec3, 8> vertices = {
            Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
            Vec3(-1, -1, 1), Vec3(1, -1, 1), Vec3(1, 1, 1), Vec3(-1, 1, 1)
        };
        for (auto& v : vertices) {
            v = v * size + position;
        }
        std::array<int, 36> indices = {
            0,1,2, 2,3,0, 4,5,6, 6,7,4, 0,4,7, 7,3,0,
            1,5,6, 6,2,1, 3,2,6, 6,7,3, 0,1,5, 5,4,0
        };
        for (size_t i = 0; i < indices.size(); i += 3) {
            Triangle tri;
            tri.vertices[0] = vertices[indices[i]];
            tri.vertices[1] = vertices[indices[i + 1]];
            tri.vertices[2] = vertices[indices[i + 2]];
            tri.color = color;
            tri.normal = (tri.vertices[1] - tri.vertices[0]).cross(tri.vertices[2] - tri.vertices[0]).normalized();
            tri.specular = specular;
            tri.reflectivity = reflectivity;
            scene.push_back(tri);
        }
        };

    addCube(Vec3(0, 0, 0), 1.0f, Vec3(0.7f, 0.2f, 0.2f), 20.0f, 0.5f);
    addCube(Vec3(-2, 0, -2), 0.7f, Vec3(0.2f, 0.7f, 0.2f), 50.0f, 0.7f);
    addCube(Vec3(2, 0, -2), 0.7f, Vec3(0.2f, 0.2f, 0.7f), 10.0f, 0.3f);
    addCube(Vec3(0, -2, -1), 0.5f, Vec3(0.7f, 0.7f, 0.2f), 30.0f, 0.6f);

    return scene;
}

int main() {
    enableANSIColors();
    const int width = 150;
    const int height = 60;
    Renderer renderer(width, height);
    std::vector<Triangle> scene = createScene();

    auto lastTime = std::chrono::high_resolution_clock::now();
    float rotation = 0.0f;
    while (true) {
        renderer.clear();

        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = currentTime - lastTime;
        float deltaTime = elapsed.count();
        lastTime = currentTime;

        Mat4 rotationMatrix = Mat4::rotation(rotation, Vec3(0, 1, 0)) *
            Mat4::rotation(sin(rotation * 0.5f) * 0.3f, Vec3(1, 0, 0)) *
            Mat4::rotation(cos(rotation * 0.3f) * 0.2f, Vec3(0, 0, 1));

        Mat4 viewMatrix = Mat4::identity();
        viewMatrix.elements[12] = sin(rotation * 0.5f) * 3.0f;
        viewMatrix.elements[13] = cos(rotation * 0.3f) * 1.5f;
        viewMatrix.elements[14] = -8.0f + sin(rotation * 0.2f) * 2.0f;

        Mat4 modelViewMatrix = viewMatrix * rotationMatrix;

        for (const auto& triangle : scene) {
            renderer.drawTriangle(triangle, modelViewMatrix);
        }

        renderer.updateParticles(deltaTime);
        renderer.drawParticles(viewMatrix);

        renderer.render();

        float fps = 1.0f / deltaTime;
        std::cout << "FPS: " << static_cast<int>(fps) << std::endl;

        rotation += 0.02f;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    return 0;
}
