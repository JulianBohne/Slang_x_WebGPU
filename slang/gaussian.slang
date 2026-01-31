module gaussian;

[Differentiable]
public float gaussianPDF(float2 coord, float2 mew, float2x2 invCov) {
    return exp(-0.5 * dot((coord - mew), mul(invCov, coord - mew)));
}

public struct SplatCamera {
    public float4x4 mv;
    public float4x4 projection;
//    float4x4 mvp;

    [Differentiable]
    [NoDiffThis]
    float3x3 jacobian(float4 at) {
        float b = projection[0][0];
        float h = projection[1][1];
        float o = projection[2][3];

        return float3x3(
            b/at.z,       0, -(b * at.x) / (at.z * at.z),
                  0, h/at.z, -(h * at.y) / (at.z * at.z),
                  0,       0,         -o / (at.z * at.z),
        );
    }

}

[ForceInline]
float trace(float2x2 mat) {
    return mat[0][0] + mat[1][1];
}

// https://math.stackexchange.com/questions/395698/fast-way-to-calculate-eigen-of-2x2-matrix-using-a-formula
// https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
float2x2 eigen(float2x2 mat) {

    float d = determinant(mat); // Note: Determinant is expected to never be negative
    float t = trace(mat);

    float l1 = t / 2 + sqrt(t * t / 4 - d);
    float l2 = t / 2 - sqrt(t * t / 4 - d);

    float2 eig_1 = normalize(float2(l1 - mat[1][1], mat[1][0])) * sqrt(l1);
    float2 eig_2 = normalize(float2(l2 - mat[1][1], mat[1][0])) * sqrt(l2);
    return float2x2(eig_1, eig_2);
}

// Rotate vec by 90 degrees
[ForceInline]
float2 rot_90(float2 vec) {
    return float2(-vec.y, vec.x);
}

public struct Gaussian2D : IDifferentiable {
    public float2 position;
    public no_diff float depth;
    public float2x2 inv_cov;

    public float3 color;
    public float alpha;

    public float2x2 circle_transformation;

    // Intersection test between the 2D gaussian and a convex quadrilateral (TODO: CC wise?)
    public bool intersects_quad(float2 quad_points[4]) {

        float2 transformed_quad_points[4];
        float2 center = float2(0, 0);
        [unroll]
        for (int i = 0; i < 4; ++i) {
            transformed_quad_points[i] = mul(circle_transformation, quad_points[i] - position);
            center += transformed_quad_points[i];
        }
        center /= 4;

        float2 testPoint = center / length(center);
        // return length(testPoint) < 1;

        float2 normals[4];
        normals[0] = normalize(rot_90(transformed_quad_points[(0 + 1) % 4] - transformed_quad_points[0]));
        normals[1] = normalize(rot_90(transformed_quad_points[(1 + 1) % 4] - transformed_quad_points[1]));
        normals[2] = normalize(rot_90(transformed_quad_points[(2 + 1) % 4] - transformed_quad_points[2]));
        normals[3] = normalize(rot_90(transformed_quad_points[(3 + 1) % 4] - transformed_quad_points[3]));
        // [unroll]
        // for (int i = 0; i < 4; ++i) {
        //     normals[i] = normalize(rot_90(transformed_quad_points[(i + 1) % 4] - transformed_quad_points[i]));
        // }
        bool intersects = dot(testPoint - transformed_quad_points[0], normals[0]) < 1
            && dot(testPoint - transformed_quad_points[1], normals[1]) < 1
            && dot(testPoint - transformed_quad_points[2], normals[2]) < 1
            && dot(testPoint - transformed_quad_points[3], normals[3]) < 1;

        bool intersects_flipped = dot(testPoint - transformed_quad_points[0], -normals[0]) < 1
            && dot(testPoint - transformed_quad_points[1], -normals[1]) < 1
            && dot(testPoint - transformed_quad_points[2], -normals[2]) < 1
            && dot(testPoint - transformed_quad_points[3], -normals[3]) < 1;
        // intersects         &= dot(testPoint - transformed_quad_points[0], normals[0]) < 1;
        // intersects_flipped &= dot(testPoint - transformed_quad_points[0], -normals[0]) < 1;
        // [unroll]
        // for (int i = 0; i < 4; ++i) {
        //     intersects &= dot(testPoint - transformed_quad_points[i], normals[i]) < 1;
        //     intersects_flipped &= dot(testPoint - transformed_quad_points[i], -normals[i]) < 1;
        // }
        return intersects || intersects_flipped;
    }
}

[Differentiable]
float3x3 quat_to_mat(float4 quaternion) {
    float4 q = normalize(quaternion);
    // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    return float3x3(
        1 - 2 * (q.y*q.y + q.z*q.z),     2 * (q.x*q.y - q.z*q.w),     2 * (q.x*q.z + q.y*q.w),
            2 * (q.x*q.y + q.z*q.w), 1 - 2 * (q.x*q.x + q.z*q.z),     2 * (q.y*q.z - q.x*q.w),
            2 * (q.x*q.z - q.y*q.w),     2 * (q.y*q.z + q.x*q.w), 1 - 2 * (q.x*q.x + q.y*q.y),
    );
}

[Differentiable]
float3x3 scale_mat(float3 scale) {
    return float3x3(
        scale.x, 0, 0,
        0, scale.y, 0,
        0, 0, scale.z,
    );
}

// Stolen from https://shader-slang.org/slang-playground/?target=SPIRV&code=eJyVU01r20AQvetXDD7Z4MqJm0OJUh_iXHoIDdgEijFlI41tkdWu2J11bUr_e2c_vJHpqUKg1czozXszT7MZrA-tBTyJrpcIDXZaWTKC0MJB_wLSHNu1CuGIhvAEQjWwM2LfoSKwB9GgAT6ac69bRbYsZgmSbwG17vpW4iet5PnSZApWAx3QoK9RGrSj3hGj6s7HoZfivDfaqSag_dAOaqHAWQzp0TKCjuDNEWnlOaY-nB-I8dzb3Y4bMVUSZo-eX8HyXE3wGvQUvwvgaye1oM_Qa9tSq1U1DCptOiEHoTm4Y1X8yUjPPC3TCpmweAwg5Bs2eo0ncga_qQZPVc5FvOvcB9iqRoUJaRVCXNU8Oq_j4dJpAV062Yib0OZPQPFkN7fzL9sh8FJ0aMRQ793pDrpjH4pehOE8Yz5KXb8_BBYLsP5RFUvvCaEosYhIC16Kf1bFKozbrIg5gR28VB_D_u7oqnUeNdzD6vXny_8NfhONNx5FU44m28J3SB59Fq0ax_XywNlak9Q61ES3RdR4LjOXr9A5OY66Sh7NNLEdB5hcN4Xb8mYyucKIZBkhlg65pwp3zFkvw2cM8q5UpjRUdvnHvLY0skso6vNaoro4wnUweFIqkS5O4KZhjWV2RnzNDtrcbMt__bqtMlCtpTYMkwDKuO_xcNPTLGxypSx86oX9Be1nf3g
[Differentiable]
float2x2 inverse(float2x2 mat) {
    float2x2 output;

    float det = determinant(mat);
    output[0][0] = mat[1][1] / det;
    output[0][1] = -mat[0][1] / det;
    output[1][0] = -mat[1][0] / det;
    output[1][1] = mat[0][0] / det;

    return output;
}

public struct Gaussian3D : IDifferentiable {
    public float3 position;
    public float4 rotation;
    public float3 scale;
    public float3 color;
    public float alpha;

    [Differentiable]
    public Gaussian2D transform(SplatCamera cam) {
        float4 cam_space_position = mul(cam.mv, float4(position, 1.0));
        float4 projected_position = mul(cam.projection, cam_space_position);
        float4 ndc_position = projected_position / projected_position.w;

        float3x3 jacobian = cam.jacobian(cam_space_position);

        float3x3 rotation_mat = quat_to_mat(rotation);
        float3x3 scale_mat = scale_mat(scale);

        float3x3 half_cov = mul(rotation_mat, scale_mat);
        
        float3x3 transformed_half_cov = mul(jacobian, mul(float3x3(cam.mv), half_cov));
        float3x3 cov = mul(transformed_half_cov, transpose(transformed_half_cov));
        float2x2 inv_cov = inverse(float2x2(cov));

        // Precalculations for intersection test
        float n_sigmas = 2;

        float2x2 half_cov_t = no_diff eigen(float2x2(cov));

        float2 axis_a = half_cov_t[0] * n_sigmas;
        float2 axis_b = half_cov_t[1] * n_sigmas;

        float2 inverse_axis_a = axis_a / dot(axis_a, axis_a);
        float2 inverse_axis_b = axis_b / dot(axis_b, axis_b);
        float2x2 circle_transformation = float2x2(inverse_axis_a, inverse_axis_b);

        return Gaussian2D(
            ndc_position.xy,
            ndc_position.z,
            inv_cov,
            color,
            alpha,
            circle_transformation,
        );
    }
}
