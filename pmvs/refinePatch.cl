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

/*
float getUnit(vec4 imCenter, float ipscale, float4 coord) {
  const float fz = length(coord - imCenter);
  const float ftmp = ipscale;
  if (ftmp == 0.0)
    return 1.0;

  return 2.0 * fz * (0x0001 << m_fm.m_level) / ftmp;
}
*/

__kernel void refinePatch(__read_only image2d_array_t images, /* 0 */
        __constant float4 *projections, /* 1 */
        float4 center, /* 2 */
        float4 ray, /* 3 */
        float dscale, /* 4 */
        float ascale, /* 5 */
        __constant float *ipscales, /* 6 */
        __constant int *indexes, /* 7 */
        __constant float3 *xaxes, /* 8 */
        __constant float3 *yaxes, /* 9 */
        __constant float3 *zaxes, /* 10 */
        __constant float4 *imCenters, /* 11 */
        __global float3 *patchVecPtr) /* 12 */ {
    float3 patchVec = patchVecPtr[0];
    float4 coord = decodeCoord(center, ray, dscale, patchVec);
    int refIdx = indexes[0];
    float4 normal = decodeNormal(xaxes[refIdx], yaxes[refIdx], zaxes[refIdx],
           ascale, patchVec);
    //float unit = getUnit(imCenters[0], ipscales[0], coord);
    patchVecPtr[0].x = normal.x;
    patchVecPtr[0].y = normal.y;
    patchVecPtr[0].z = normal.z;
}
