module gaussian;

[Differentiable]
public float gaussianPDF(float2 coord, float2 mew, float2x2 invCov) {
    return exp(-0.5 * dot((coord - mew), mul(invCov, coord - mew)));
}

public struct SplatCamera {
    float4x4 mv;
    float4x4 projection;
//    float4x4 mvp;

    [Differentiable]
    [NoDiffThis]
    float3x3 jacobian(float4 at) {
        // TODO: Double chjeck this lol
        float a = projection[0][0];
        float b = projection[1][1];
        float d = projection[3][2];

        return transpose(float3x3(
            -a/at.z,       0, (a * at.x) / (at.z * at.z),
                  0, -b/at.z, (b * at.y) / (at.z * at.z),
                  0,       0,          d / (at.z * at.z),
        ));
    }

}

public struct Gaussian2D : IDifferentiable {
    public float2 position;
    public no_diff float depth;
    public float2x2 inv_cov;

    public float3 color;
    public float alpha;

    public no_diff float2 axis_a;
    public no_diff float2 axis_b;
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
    float3 position;
    float4 rotation;
    float3 scale;
    float3 color;
    float alpha;

    [Differentiable]
    public Gaussian2D transform(SplatCamera cam) {
        float4 cam_space_position = mul(cam.mv, float4(this.position, 1.0));
        float4 projected_position = mul(cam.projection, cam_space_position);
        float4 ndc_position = projected_position / projected_position.w;

        float3x3 jacobian = cam.jacobian(cam_space_position);
        float3x3 rotation_mat = quat_to_mat(rotation);
        float3x3 scale_mat = scale_mat(scale);
        float3x3 half_cov = mul(rotation_mat, scale_mat);
        float3x3 transformed_half_cov = mul(jacobian, mul(float3x3(cam.mv), half_cov));
        float3x3 cov = mul(transformed_half_cov, transpose(transformed_half_cov));
        float2x2 inv_cov = inverse(float2x2(cov));
        // float2x2 inv_half_cov = inverse(float2x2(transformed_half_cov));

        return Gaussian2D(
            ndc_position.xy,
            ndc_position.z,
            inv_cov,
            color,
            alpha,
            float2(transformed_half_cov[0][0], transformed_half_cov[1][0]),
            float2(transformed_half_cov[0][1], transformed_half_cov[1][1]),
        );
    }
}
