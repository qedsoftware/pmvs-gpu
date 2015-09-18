const sampler_t pmvsSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

float4 decodeCoord(float4 center, float4 ray, float dscale, float3 patchVec) {
    return center + dscale * patchVec.x * ray;
}

float4 decodeNormal(float3 xaxis, float3 yaxis, float3 zaxis, float ascale, float3 patchVec) {
    float4 rval;
    float angle1 = patchVec.y * ascale;
    float angle2 = patchVec.z * ascale;

    float fx = sin(angle1) * cos(angle2);
    float fy = sin(angle2);
    float fz = - cos(angle1) * cos(angle2);

    float3 ftmp = xaxis * fx + yaxis * fy + zaxis * fz;
    rval.x = ftmp.x;
    rval.y = ftmp.y;
    rval.z = ftmp.z;
    rval.w = 0.0f;
    return rval;
}

__kernel void refinePatch(__read_only image2d_array_t images, /* 0 */
        __constant float4 *projections, /* 1 */
        float4 center, /* 2 */
        float4 ray, /* 3 */
        float dscale, /* 4 */
        float ascale, /* 5 */
        __constant int *indexes, /* 6 */
        __constant float3 *xaxes, /* 7 */
        __constant float3 *yaxes, /* 8 */
        __constant float3 *zaxes, /* 9 */
        __global float3 *patchVecPtr) /* 10 */ {
    float3 patchVec = patchVecPtr[0];
    float4 coord = decodeCoord(center, ray, dscale, patchVec);
    int refIdx = indexes[0];
    float4 normal = decodeNormal(xaxes[refIdx], yaxes[refIdx], zaxes[refIdx],
           ascale, patchVec);
    patchVecPtr[0].x = normal.x;
    patchVecPtr[0].y = normal.y;
    patchVecPtr[0].z = normal.z;
}
