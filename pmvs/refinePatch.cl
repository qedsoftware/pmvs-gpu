#define WSIZE <WSIZE>
const sampler_t imSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

typedef struct _ImageParams {
    float4 projection[3];
    float3 xaxis;
    float3 yaxis;
    float3 zaxis;
    float4 center;
    float ipscale;
} ImageParams;

typedef struct _PatchParams {
    float4 center;
    float4 ray;
    float dscale;
    float ascale;
    int nIndexes;
    int indexes[10];
} PatchParams;

struct FArgs {
    PatchParams patch;
    __constant ImageParams *images;
    int level;
    __local float *refData;
    __local float *imData;
    __local float *localVal;
};

float4 decodeCoord(float4 center, float4 ray, float dscale, double3 patchVec) {
    double4 dray = (double4)(ray.x, ray.y, ray.z, ray.w);
    return center + convert_float4((double)dscale * patchVec.x * dray);
}

float4 decodeNormal(float3 xaxis, float3 yaxis, float3 zaxis, float ascale, double3 patchVec) {
    float4 rval;
    float angle1 = (float)patchVec.y * ascale;
    float angle2 = (float)patchVec.z * ascale;

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

void imNormalize(__local float *texData, float4 cavg) {
    float std=0.f;
    float3 cnorm;
    __local float *ctexData = texData-1;
    for(int i=0; i<WSIZE*WSIZE; i++) {
        cnorm.x = *(++ctexData) - cavg.x;
        cnorm.y = *(++ctexData) - cavg.y;
        cnorm.z = *(++ctexData) - cavg.z;
        std += dot(cnorm, cnorm);
    }
    std = sqrt(std/(WSIZE*WSIZE*3));

    ctexData = texData-1;
    for(int i=0; i<WSIZE*WSIZE; i++) {
        *(++ctexData) -= cavg.x;  *ctexData /= std;
        *(++ctexData) -= cavg.y;  *ctexData /= std;
        *(++ctexData) -= cavg.z;  *ctexData /= std;
    }
}

int grabTex(__read_only image2d_array_t images,
        int imIndex,
        __constant float4 *projection,
        float4 imCenter,
        float4 coord,
        float4 pxaxis,
        float4 pyaxis,
        float4 pzaxis,
        __local float *texData) {
    float4 cavg = 0.f;
    float4 ray = normalize(imCenter - coord);
    size_t localX = get_local_id(0);
    size_t localY = get_local_id(1);
    
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
    /*
    for (int y = 0; y < WSIZE; ++y) {
      float3 vftmp = left;
      left += dy;
      for (int x = 0; x < WSIZE; ++x) {
        imCoord.x = vftmp.x+.5;
        imCoord.y = vftmp.y+.5;
        float4 color = read_imagef(images, imSampler, imCoord);
        *(++texp) = color.x;
        *(++texp) = color.y;
        *(++texp) = color.z;
        vftmp += dx;
      }
    }
    */
    float3 imCoord3 = left + dx * localX + dy * localY;
    imCoord.x = imCoord3.x + .5;
    imCoord.y = imCoord3.y + .5;
    float4 color = read_imagef(images, imSampler, imCoord);
    texData[localY*WSIZE*3 + localX*3] = color.x;
    texData[localY*WSIZE*3 + localX*3 + 1] = color.y;
    texData[localY*WSIZE*3 + localX*3 + 2] = color.z;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localX == 0 && localY == 0) {
      texp = texData - 1;
      for (int i = 0; i < WSIZE*WSIZE; ++i) {
        cavg.x += *(++texp);
        cavg.y += *(++texp);
        cavg.z += *(++texp);
      }
      cavg /= (WSIZE*WSIZE);
      imNormalize(texData, cavg);
    }
    return 0;
}

float robustincc(const float rhs) {
    return rhs / (1 + 3 * rhs);
}

float evalF(double3 patchVec,
        __read_only image2d_array_t images,
        struct FArgs *args) {
    size_t localX = get_local_id(0);
    size_t localY = get_local_id(1);
    float4 coord = decodeCoord(args->patch.center, args->patch.ray, args->patch.dscale, patchVec);
    int refIdx = args->patch.indexes[0];
    float4 normal = decodeNormal(args->images[refIdx].xaxis, args->images[refIdx].yaxis, args->images[refIdx].zaxis,
           args->patch.ascale, patchVec);
    float pscale = getUnit(args->images[refIdx].center, args->images[refIdx].ipscale, coord, args->level);
    float3 normal3 = (float3)(normal.x, normal.y, normal.z);
    float3 yaxis3 = normalize(cross(normal3, args->images[refIdx].xaxis));
    float3 xaxis3 = cross(yaxis3, normal3);
    __constant float4 *refProj = args->images[refIdx].projection;
    float4 pxaxis = getPAxis(refProj, coord, normal, xaxis3, pscale);
    float4 pyaxis = getPAxis(refProj, coord, normal, yaxis3, pscale);
    grabTex(images, refIdx, refProj, args->images[refIdx].center, coord, pxaxis, pyaxis, normal, args->refData);
    float ans = 0.;
    int denom = 0;
    for(int i=1; i < args->patch.nIndexes; i++) {
        int imIdx = args->patch.indexes[i];
        __constant float4 *imProj = args->images[imIdx].projection;
        grabTex(images, imIdx, imProj, args->images[imIdx].center, coord, pxaxis, pyaxis, normal, args->imData);
        float corr = 0.;
        if(localX == 0 && localY == 0) {
            for(int j=0; j<WSIZE*WSIZE*3; j++) {
                corr += args->refData[j] * args->imData[j];
            }
            ans += robustincc(1.-corr/(WSIZE*WSIZE*3));
            denom++;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(localX == 0 && localY == 0) {
        *(args->localVal) = ans / denom;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return *(args->localVal);
}

double3 testStep(double3 patchVec, double3 step, __read_only image2d_array_t images, struct FArgs *args, 
        float *val, bool *didStep, float *maxDiff) {
    size_t localX = get_local_id(0);
    size_t localY = get_local_id(1);
    double3 test1 = patchVec+step;
    double3 test2 = patchVec-step;
    float newval = evalF(test1, images, args);
    float diff = fabs(newval-*val);
    if(diff > *maxDiff) *maxDiff = diff;
    if(newval < *val) {
        patchVec = test1;
        *val = newval;
        *didStep = true;
    }
    newval = evalF(test2, images, args);
    diff = fabs(newval-*val);
    if(diff > *maxDiff) *maxDiff = diff;
    if(newval < *val) {
        patchVec = test2;
        *val = newval;
        *didStep = true;
    }
    return patchVec;
}

__kernel void refinePatch(__read_only image2d_array_t images, /* 0 */
        __constant ImageParams *imageParams, /* 1 */
        __global PatchParams *patchParams, /* 2 */
        int level, /* 3 */
        __global double4 *encodedVecs) /* 4 */
{
    __local float refData[3*WSIZE*WSIZE];
    __local float imData[3*WSIZE*WSIZE];
    __local float localVal;

    size_t groupId = get_group_id(0);

    __global PatchParams *myPatchParams = patchParams + groupId;
    double4 encodedVec = encodedVecs[groupId];
    double3 patchVec = (double3)(encodedVec.x, encodedVec.y, encodedVec.z);

    struct FArgs args;
    args.images = imageParams;
    args.patch = *myPatchParams;
    args.level = level;
    args.refData = refData;
    args.imData = imData;
    args.localVal = &localVal;

    int maxSteps = 50;
    double3 stepX = (double3)(.1,0,0);
    double3 stepY = (double3)(0,.75,0);
    double3 stepZ = (double3)(0,0,.75);
    int cstep = 0;
    int nreduce = 0;
    bool didStep;
    float val = evalF(patchVec, images, &args);
    if(encodedVec.w >= 0) {
        while(cstep < maxSteps) {
            float maxDiff = 0;
            float oldval = val;
            didStep = false;
            patchVec = testStep(patchVec, stepX, images, &args, &val, &didStep, &maxDiff);
            patchVec = testStep(patchVec, stepY, images, &args, &val, &didStep, &maxDiff);
            patchVec = testStep(patchVec, stepZ, images, &args, &val, &didStep, &maxDiff);
            if(!didStep) {
                stepX /= 4.;
                stepY /= 4.;
                stepZ /= 4.;
                nreduce++;
            }
            cstep++;
            //if(nreduce > 3 && fabs(oldval-val) < .001f) break;
            encodedVec.w = maxDiff;
        }
    }
    //float val = evalF(images, projections, center, ray, dscale,
    //    ascale, ipscales, indexes, xaxes, yaxes, zaxes,
    //    imCenters, patchVecPtr[0], level, nIndexes);
    //float val = evalF(patchVecPtr[0], images, &args);

    if(get_local_id(0) == 0 && get_local_id(1) == 0) {
        encodedVecs[groupId].x = patchVec.x;
        encodedVecs[groupId].y = patchVec.y;
        encodedVecs[groupId].z = patchVec.z;
        encodedVecs[groupId].w = encodedVec.w;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}
