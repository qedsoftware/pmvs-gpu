#define WSIZE <WSIZE>
#define M_TAU_MAX <M_TAU_MAX>
const sampler_t imSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

enum {
    SIMPLEX_STATE_INIT_ALL = 0,
    SIMPLEX_STATE_INIT,
    SIMPLEX_STATE_REFLECT,
    SIMPLEX_STATE_EXPAND,
    SIMPLEX_STATE_CONTRACT,
    SIMPLEX_STATE_FAILED_CONTRACT
};

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
    int indexes[M_TAU_MAX];
} PatchParams;

struct FArgs {
    PatchParams *patch;
    __constant ImageParams *images;
    int level;
};

float4 decodeCoord(float4 center, float4 ray, float dscale, float3 patchVec) {
    float4 dray = (float4)(ray.x, ray.y, ray.z, ray.w);
    return center + convert_float4((float)dscale * patchVec.x * dray);
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

float2 project2(__constant float4 *projections, float4 coord) {
    float3 p3 = project(projections, coord);
    return (float2)(p3.x, p3.y);
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

/*
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
    size_t localX = get_local_id(0);
    size_t localY = get_local_id(1);
    
    __local float* texp = texData - 1;
    float4 imCoord;
    imCoord.z = imIndex;
    imCoord.w = 0.f;
    //for (int y = 0; y < WSIZE; ++y) {
    //  float3 vftmp = left;
    //  left += dy;
    //  for (int x = 0; x < WSIZE; ++x) {
    //    imCoord.x = vftmp.x+.5;
    //    imCoord.y = vftmp.y+.5;
    //    float4 color;
    //    color = read_imagef(images, imSampler, imCoord);
    //    *(++texp) = color.x;
    //    *(++texp) = color.y;
    //    *(++texp) = color.z;
    //    vftmp += dx;
    //  }
    //}

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
*/

float3 computeTexParams(__constant float4 *projection,
        float4 coord,
        float4 pxaxis,
        float4 pyaxis,
        float2 *begin,
        float2 *dx,
        float2 *dy) {
    //const float weight = max(0.0f, dot(ray, pzaxis));

    //if (weight < cos(m_fm.m_angleThreshold1))
    //  return 1;

    const int margin = WSIZE / 2;

    float2 center = project2(projection, coord);
    *dx = project2(projection, coord + pxaxis) - center;
    *dy = project2(projection, coord + pyaxis) - center;

    // TODO leveldif

    *begin = center - (*dx) * margin - (*dy) * margin;
}

float3 texAvg(float2 *begin, float2 *dx, float2 *dy, __read_only image2d_array_t images, int imIndex) {
    float2 left = *begin;
    float2 ccoord;
    float4 imCoord;
    imCoord.z = imIndex;
    imCoord.w = 0.f;
    float3 t = (float3)(0,0,0);
    for(int y=0; y < WSIZE; y++) {
        ccoord = left;
        left += *dy;
        for(int x=0; x < WSIZE; x++) {
            imCoord.x = ccoord.x+.5;
            imCoord.y = ccoord.y+.5;
            float4 color = read_imagef(images, imSampler, imCoord);
            t += as_float3(color);
            ccoord += *dx;
        }
    }
    return t / (WSIZE*WSIZE);
}

float texStd(float2 *begin, float2 *dx, float2 *dy, float3 *avg, __read_only image2d_array_t images, int imIndex) {
    float2 left = *begin;
    float2 ccoord;
    float4 imCoord;
    imCoord.z = imIndex;
    imCoord.w = 0.f;
    float3 cnorm;
    float std=0.f;
    for(int y=0; y < WSIZE; y++) {
        ccoord = left;
        left += *dy;
        for(int x=0; x < WSIZE; x++) {
            imCoord.x = ccoord.x+.5;
            imCoord.y = ccoord.y+.5;
            float4 color = read_imagef(images, imSampler, imCoord);
            cnorm = as_float3(color) - *avg;
            std += dot(cnorm, cnorm);
            ccoord += *dx;
        }
    }
    return sqrt(std/(WSIZE*WSIZE*3));
}

float robustincc(const float rhs) {
    return rhs / (1 + 3 * rhs);
}

float evalF(float3 patchVec,
        __read_only image2d_array_t images,
        struct FArgs *args) {
    float4 coord = decodeCoord(args->patch->center, args->patch->ray, args->patch->dscale, patchVec);
    int refIdx = args->patch->indexes[0];
    float4 normal = decodeNormal(args->images[refIdx].xaxis, args->images[refIdx].yaxis, args->images[refIdx].zaxis,
           args->patch->ascale, patchVec);
    float pscale = getUnit(args->images[refIdx].center, args->images[refIdx].ipscale, coord, args->level);
    float3 normal3 = (float3)(normal.x, normal.y, normal.z);
    float3 yaxis3 = normalize(cross(normal3, args->images[refIdx].xaxis));
    float3 xaxis3 = cross(yaxis3, normal3);
    __constant float4 *refProj = args->images[refIdx].projection;
    float4 pxaxis = getPAxis(refProj, coord, normal, xaxis3, pscale);
    float4 pyaxis = getPAxis(refProj, coord, normal, yaxis3, pscale);
    float2 refBegin;
    float2 refDX;
    float2 refDY;
    computeTexParams(refProj, coord, pxaxis, pyaxis, &refBegin, &refDX, &refDY);
    float3 refAvg = texAvg(&refBegin, &refDX, &refDY, images, refIdx);
    float refStd = texStd(&refBegin, &refDX, &refDY, &refAvg, images, refIdx);
    float ans=0.f;
    int denom=0;
    for(int i=1; i < args->patch->nIndexes; i++) {
        int imIdx = args->patch->indexes[i];
        float2 imBegin, imDX, imDY;
        computeTexParams(args->images[imIdx].projection, coord, pxaxis, pyaxis, &imBegin, &imDX, &imDY);
        float3 imAvg = texAvg(&imBegin, &imDX, &imDY, images, imIdx);
        float imStd = texStd(&imBegin, &imDX, &imDY, &imAvg, images, imIdx);
        float2 imLeft = imBegin;
        float2 refLeft = refBegin;
        float4 imCoord, refCoord;
        imCoord.z = imIdx;
        imCoord.w = 0.f;
        refCoord.z = refIdx;
        refCoord.w = 0.f;
        float2 imc, refc;
        float corr = 0.f;
        for(int y=0; y<WSIZE; y++) {
            imc = imLeft;
            refc = refLeft;
            imLeft += imDY;
            refLeft += refDY;
            for(int x=0; x<WSIZE; x++) {
                imCoord.x = imc.x + .5;
                imCoord.y = imc.y + .5;
                refCoord.x = refc.x + .5;
                refCoord.y = refc.y + .5;
                float3 imColor = as_float3(read_imagef(images, imSampler, imCoord));
                float3 refColor = as_float3(read_imagef(images, imSampler, refCoord));
                corr += dot((imColor - imAvg) / imStd, (refColor - refAvg) / refStd);
                imc += imDX;
                refc += refDX;
            }
        }
        ans += robustincc(1.-corr/(WSIZE*WSIZE*3));
        denom++;
    }
    return ans / denom;
}

void initSimplexVec(int i, float4 *simplexVecs, float3 vec) {
    *((float3 *)(simplexVecs+i)) = vec;
    simplexVecs[i].w = -1;
}

void setSimplexVec(int i, float4 *simplexVecs, float3 vec, float val) {
    *((float3 *)(simplexVecs+i)) = vec;
    simplexVecs[i].w = val;
}

float3 simplexReflect(float coeff, float4 *simplexVecs, int hi) {
    float3 v = (float3)(0,0,0);
    for(int i=0; i<4; i++) {
        if(i==hi) continue;
        v += as_float3(simplexVecs[i]);
    }
    v /= 3.f;
    v = v - coeff * (v - as_float3(simplexVecs[hi]));
    return v;
}

void simplexMoveTowardsBest(float4 *simplexVecs, int lo) {
    float3 vlo = as_float3(simplexVecs[lo]);
    float3 v;
    for(int i=0; i<4; i++) {
        if(i==lo) continue;
        v = (as_float3(simplexVecs[i]) + vlo) / 2.f;
        setSimplexVec(i, simplexVecs, v, -1);
    }
}

float simplexFVariance(float4 *simplexVecs) {
    float m=0;
    for(int i=0; i<4; i++) {
        m += simplexVecs[i].w;
    }
    m /= 4.f;
    float t=0;
    for(int i=0; i<4; i++) {
        t += pow(simplexVecs[i].w-m, 2.f);
    }
    return sqrt(t);
}

float simplexSize(float4 *simplexVecs) {
    float3 center = (float3)(0,0,0);
    for(int i=0; i<4; i++) {
        center += as_float3(simplexVecs[i]);
    }
    center /= 4.f;
    float t=0;
    for(int i=0; i<4; i++) {
        t += length(center-as_float3(simplexVecs[i]));
    }
    return t/4.;
}

__kernel void refinePatch(__read_only image2d_array_t images, /* 0 */
        __constant ImageParams *imageParams, /* 1 */
        __global PatchParams *patchParams, /* 2 */
        int level, /* 3 */
        __global float4 *encodedVecs, /* 4 */
        __global float4 *simplexVecs, /* 5 */
        __global int *simplexStates) /* 6 */
{
    float4 mySimplexVecs[4];
    PatchParams myPatchParams;

    size_t globalId = get_global_id(0);

    myPatchParams = patchParams[globalId];
    float4 encodedVec = encodedVecs[globalId];
    float3 patchVec = (float3)(encodedVec.x, encodedVec.y, encodedVec.z);

    struct FArgs args;
    args.images = imageParams;
    args.patch = &myPatchParams;
    args.level = level;

    int maxSteps = 10;
    float3 stepX = (float3)(1.,0,0);
    float3 stepY = (float3)(0,1.,0);
    float3 stepZ = (float3)(0,0,1.);

    int state;

    state = simplexStates[globalId];
    if(state == SIMPLEX_STATE_INIT_ALL) {
        initSimplexVec(0, mySimplexVecs, patchVec);
        initSimplexVec(1, mySimplexVecs, patchVec+stepX);
        initSimplexVec(2, mySimplexVecs, patchVec+stepY);
        initSimplexVec(3, mySimplexVecs, patchVec+stepZ);
        state = SIMPLEX_STATE_INIT;
    }
    else {
        for(int i=0; i<4; i++) {
            mySimplexVecs[i] = simplexVecs[4*globalId+i];
        }
    }

    int cstep=0;
    int hi, s_hi, lo=0, initIdx=0;
    float dhi, ds_hi, dlo;
    float val, val2;
    float3 testVec, testVecLast;

    if(encodedVec.w >= 0) {
        while(cstep < maxSteps) {
            // find next point to eval
            if(state == SIMPLEX_STATE_INIT) {
                int i=initIdx;
                initIdx = -1;
                for(; i<4; i++) {
                    if(mySimplexVecs[i].w < 0) {
                        initIdx = i;
                        break;
                    }
                }
                if(initIdx==-1) {
                    state = SIMPLEX_STATE_REFLECT;
                }
                else {
                    testVec = as_float3(mySimplexVecs[initIdx]);
                }
            }
            if(state == SIMPLEX_STATE_REFLECT) {
                dhi = dlo = mySimplexVecs[0].w;
                hi = lo = 0;
                ds_hi = mySimplexVecs[1].w;
                s_hi = 1;

                for(int i=1; i<4; i++) {
                    val = mySimplexVecs[i].w;
                    if(val < dlo) {
                        dlo = val;
                        lo = i;
                    }
                    else if(val > dhi) {
                        ds_hi = dhi;
                        s_hi = hi;
                        dhi = val;
                        hi = i;
                    }
                    else if(val > ds_hi) {
                        ds_hi = val;
                        s_hi = i;
                    }
                }
                testVec = simplexReflect(-1., mySimplexVecs, hi);
            }
            else if(state == SIMPLEX_STATE_EXPAND) {
                testVec = simplexReflect(-2, mySimplexVecs, hi);
            }
            else if(state == SIMPLEX_STATE_CONTRACT) {
                testVec = simplexReflect(.5, mySimplexVecs, hi);
            }
        
            // branches converged, send all threads in warp to evalF
            val = evalF(testVec, images, &args);

            // figure out new state
            if(state == SIMPLEX_STATE_INIT) {
                setSimplexVec(initIdx, mySimplexVecs, testVec, val);
                initIdx++;
            }
            else if(state == SIMPLEX_STATE_REFLECT) {
                if(val < mySimplexVecs[lo].w) {
                    testVecLast = testVec;
                    val2 = val;
                    state = SIMPLEX_STATE_EXPAND;
                }
                else if(val > mySimplexVecs[s_hi].w) {
                    if(val < mySimplexVecs[hi].w) {
                        setSimplexVec(hi, mySimplexVecs, testVec, val);
                    }
                    state = SIMPLEX_STATE_CONTRACT;
                }
                else {
                    setSimplexVec(hi, mySimplexVecs, testVec, val);
                }
            }
            else if(state == SIMPLEX_STATE_EXPAND) {
                if(val < mySimplexVecs[lo].w) {
                    setSimplexVec(hi, mySimplexVecs, testVec, val);
                }
                else {
                    setSimplexVec(hi, mySimplexVecs, testVecLast, val2);
                }
                state = SIMPLEX_STATE_REFLECT;
            }
            else if(state == SIMPLEX_STATE_CONTRACT) {
                if(val <= mySimplexVecs[hi].w) {
                    setSimplexVec(hi, mySimplexVecs, testVec, val);
                    state = SIMPLEX_STATE_REFLECT;
                }
                else {
                    simplexMoveTowardsBest(mySimplexVecs, lo);
                    state = SIMPLEX_STATE_INIT;
                    initIdx = 0;
                }
            }
            cstep++;
        }
    }

    // store global state
    encodedVec = mySimplexVecs[lo];
    encodedVecs[globalId].x = encodedVec.x;
    encodedVecs[globalId].y = encodedVec.y;
    encodedVecs[globalId].z = encodedVec.z;
    //encodedVecs[globalId].w = simplexFVariance(mySimplexVecs);
    encodedVecs[globalId].w = simplexSize(mySimplexVecs);
    for(int i=0; i<4; i++) {
        simplexVecs[4*globalId+i] = mySimplexVecs[i];
    }
    if(state == SIMPLEX_STATE_EXPAND || state == SIMPLEX_STATE_CONTRACT) {
        // have to redo an iteration in this case
        // only way to avoid it is to store more global state
        state = SIMPLEX_STATE_REFLECT;
    }
    simplexStates[globalId] = state;
}
