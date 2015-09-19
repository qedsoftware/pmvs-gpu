#define WSIZE <WSIZE>
const sampler_t imSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

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

float getUnit(float4 imCenter, float ipscale, float4 coord, int level) {
  const float fz = length(coord - imCenter);
  const float ftmp = ipscale;
  if (ftmp == 0.0)
    return 1.0;

  return 2.0 * fz * (0x0001 << level) / ftmp;
}

float3 project(__constant float4 *projections, float4 coord) {
    float3 vtmp;
    vtmp.x = dot(projections[0], coord);
    vtmp.y = dot(projections[1], coord);
    vtmp.z = dot(projections[2], coord);
    vtmp /= vtmp.z;
    return vtmp;
}

float4 getPAxis(__constant float4 *projections, float4 coord, float4 normal, float3 axis3, float pscale) {
    float4 paxis;
    paxis.x = axis3.x;
    paxis.y = axis3.y;
    paxis.z = axis3.z;
    paxis.w = 0.0f;
    paxis *= pscale;

    float dis = length(project(projections, coord + paxis) -
            project(projections, coord));
    return paxis / dis;
}

int grabTex(__read_only image2d_array_t images,
        int imIndex,
        __constant float4 *projection,
        float4 ray,
        float4 coord,
        float4 pxaxis,
        float4 pyaxis,
        float4 pzaxis,
        __local float *texData) {
    const float weight = max(0.0f, dot(ray, pzaxis));

    //if (weight < cos(m_fm.m_angleThreshold1))
    //  return 1;

    const int margin = WSIZE / 2;

    float3 center = project(projection, coord);
    float3 dx = project(projection, coord + pxaxis) - center;
    float3 dy = project(projection, coord + pyaxis) - center;

    // TODO leveldif

    float3 left = center - dx * margin - dy * margin;

    __local float* texp = texData - 1;
    float4 imCoord;
    imCoord.z = imIndex;
    imCoord.w = 0.f;
    for (int y = 0; y < WSIZE; ++y) {
      float3 vftmp = left;
      left += dy;
      for (int x = 0; x < WSIZE; ++x) {
        imCoord.x = vftmp.x;
        imCoord.y = vftmp.y;
        float4 color = read_imagef(images, imSampler, imCoord);
        *(++texp) = color.x;
        *(++texp) = color.y;
        *(++texp) = color.z;
        vftmp += dx;
      }
    }

    return 0;
}

float evalF(__read_only image2d_array_t images, /* 0 */
        __constant float4 *projections,
        float4 center,
        float4 ray, /* 3 */
        float dscale, /* 4 */
        float ascale, /* 5 */
        __constant float *ipscales, /* 6 */
        __constant int *indexes, /* 7 */
        __constant float3 *xaxes, /* 8 */
        __constant float3 *yaxes, /* 9 */
        __constant float3 *zaxes, /* 10 */
        __constant float4 *imCenters, /* 11 */
        float3 patchVec, /* 12 */ 
        int level, /* 13 */
        int nIndexes) /* 14 */
{
    __local float refData[3*WSIZE*WSIZE];
    float4 coord = decodeCoord(center, ray, dscale, patchVec);
    int refIdx = indexes[0];
    float4 normal = decodeNormal(xaxes[refIdx], yaxes[refIdx], zaxes[refIdx],
           ascale, patchVec);
    float pscale = getUnit(imCenters[refIdx], ipscales[refIdx], coord, level);
    float3 normal3 = (float3)(normal.x, normal.y, normal.z);
    float3 yaxis3 = normalize(cross(normal3, xaxes[refIdx]));
    float3 xaxis3 = cross(yaxis3, normal3);
    __constant float *refProj = projections + 3*refIdx;
    float4 pxaxis = getPAxis(refProj, coord, normal, xaxis3, pscale);
    float4 pyaxis = getPAxis(refProj, coord, normal, yaxis3, pscale);
    grabTex(images, refIdx, refProj, ray, coord, pxaxis, pyaxis, normal, refData);
    return refData[10]*255.;
}

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
        __global float3 *patchVecPtr, /* 12 */ 
        int level, /* 13 */
        int nIndexes) /* 14 */
{
    float val = evalF(images, projections, center, ray, dscale,
        ascale, ipscales, indexes, xaxes, yaxes, zaxes,
        imCenters, patchVecPtr[0], level, nIndexes);

    patchVecPtr[0].x = val;
    patchVecPtr[0].y = val;
    patchVecPtr[0].z = val;
}
