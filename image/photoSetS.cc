#include <fstream>
#include <algorithm>
#include "photoSetS.h"

using namespace std;
using namespace Image;

CphotoSetS::CphotoSetS(void) {
    m_clImageArray = NULL;
    m_clImageProjections = NULL;
}

CphotoSetS::~CphotoSetS() {
}

void CphotoSetS::init(const std::vector<int>& images, const std::string prefix,
                      const int maxLevel, const int size, const int alloc) {
  initCL();
  m_images = images;
  m_num = (int)images.size();
  
  for (int i = 0; i < (int)images.size(); ++i)
    m_dict[images[i]] = i;

  int maxWidth = 0;
  int maxHeight = 0;
  m_prefix = prefix;
  m_maxLevel = max(1, maxLevel);
  m_photos.resize(m_num);
  cerr << "Reading images: " << flush;
  for (int index = 0; index < m_num; ++index) {
    const int image = m_images[index];

    char test0[1024], test1[1024];
    sprintf(test0, "%svisualize/%08d.ppm", prefix.c_str(), image);
    sprintf(test1, "%svisualize/%08d.jpg", prefix.c_str(), image);
    if (ifstream(test0) || ifstream(test1)) {
      char name[1024], mname[1024], ename[1024], cname[1024];    
      
      // Set name
      sprintf(name, "%svisualize/%08d", prefix.c_str(), image);
      sprintf(mname, "%smasks/%08d", prefix.c_str(), image);
      sprintf(ename, "%sedges/%08d", prefix.c_str(), image);
      sprintf(cname, "%stxt/%08d.txt", prefix.c_str(), image);
      
      m_photos[index].init(name, mname, ename, cname, m_maxLevel);        
      if (alloc) {
        m_photos[index].alloc();
        unsigned char *imData = m_photos[index].imData();
      }
      else
        m_photos[index].alloc(1);
      cerr << '*' << flush;
    }
    // try 4 digits
    else {
      char name[1024], mname[1024], ename[1024], cname[1024];    
      
      // Set name
      sprintf(name, "%svisualize/%04d", prefix.c_str(), image);
      sprintf(mname, "%smasks/%04d", prefix.c_str(), image);
      sprintf(ename, "%sedges/%04d", prefix.c_str(), image);
      sprintf(cname, "%stxt/%04d.txt", prefix.c_str(), image);
      
      m_photos[index].init(name, mname, ename, cname, m_maxLevel);        
      if (alloc)
        m_photos[index].alloc();
      else
        m_photos[index].alloc(1);
      cerr << '*' << flush;
    }

    if(m_photos[index].getWidth() > maxWidth) {
        maxWidth = m_photos[index].getWidth();
    }
    if(m_photos[index].getHeight() > maxHeight) {
        maxHeight = m_photos[index].getHeight();
    }

    /*
    const int image = m_images[index];
    char name[1024], mname[1024], ename[1024], cname[1024];

    // Set name
    sprintf(name, "%svisualize/%08d", prefix.c_str(), image);
    sprintf(mname, "%smasks/%08d", prefix.c_str(), image);
    sprintf(ename, "%sedges/%08d", prefix.c_str(), image);
    sprintf(cname, "%stxt/%08d.txt", prefix.c_str(), image);

    m_photos[index].init(name, mname, ename, cname, m_maxLevel);        
    if (alloc)
      m_photos[index].alloc();
    else
      m_photos[index].alloc(1);
    cerr << '*' << flush;
    */
  }
  cerr << endl;
  const int margin = size / 2;
  m_size = 2 * margin + 1;

  printf("maxWidth %d maxHeight %d\n", maxWidth, maxHeight);

  if(alloc) {
      cl_int clErr;
      cl_image_format imFormat = {CL_RGBA, CL_UNORM_INT8};
      unsigned char *rgbaBuffer = (unsigned char *)malloc(maxWidth*maxHeight*4);
      cl_image_desc imDesc = {
          CL_MEM_OBJECT_IMAGE2D_ARRAY,
          maxWidth, maxHeight, 1, m_num,
          0, 0, 0, 0, NULL};

      m_clImageArray = clCreateImage(m_clCtx,
              CL_MEM_READ_ONLY,
              &imFormat,
              &imDesc, 
              NULL,
              &clErr);
      printf("created CL image array %x\n", clErr);

      cl_command_queue clQueue = clCreateCommandQueue(m_clCtx, m_clDevice, 0, &clErr);

      for(int i=0; i<m_num; i++) {
          int imWidth = m_photos[i].getWidth();
          int imHeight = m_photos[i].getHeight();
          // must convert to RGBA because nvidia doesn't support RGB
          rgbToRGBA(imWidth, imHeight, m_photos[i].imData(), rgbaBuffer);
          size_t origin[] = {0,0,i};
          size_t region[] = {imWidth, imHeight, 1};
          clEnqueueReadImage(clQueue, m_clImageArray, CL_FALSE,
                  origin, region, 0, 0,
                  rgbaBuffer, NULL, 0, NULL);
      }

      size_t projDataSize = m_num * 3 * 4 * sizeof(float);
      float *imProjectionData = (float *)malloc(projDataSize);
      float *cptr = imProjectionData;
      for(int i=0; i<m_num; i++) {
          for(int j=0; j<3; j++) {
              for(int k=0; k<4; k++) {
                  *cptr = m_photos[i].m_projection[0][j][k];
                  cptr++;
              }
          }
      }
      m_clImageProjections = clCreateBuffer(m_clCtx, CL_MEM_READ_ONLY, 
              projDataSize, imProjectionData, &clErr);
      free(imProjectionData);

      clFinish(clQueue);
      free(rgbaBuffer);
      clReleaseCommandQueue(clQueue);
  }
  
}

void CphotoSetS::initCL() {
    cl_uint numPlatforms, numDevices;
    cl_int cl_err;
    cl_platform_id platforms[1];
    cl_device_id devices[1];
    clGetPlatformIDs(1, platforms, &numPlatforms);
    const cl_context_properties cl_props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0};
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, devices, &numDevices);
    m_clCtx = clCreateContext(cl_props, 1, devices, NULL, NULL, &cl_err);
    if(cl_err == CL_SUCCESS) {
        printf("OpenCL context created successfully\n");
    }
    else {
        printf("OpenCL error creating context %d\n", cl_err);
    }
    m_clDevice = devices[0];
}

void CphotoSetS::freePhotoSet(void) {
  for (int index = 0; index < (int)m_photos.size(); ++index)
    m_photos[index].free();
  if(m_clImageArray != NULL) {
      clReleaseMemObject(m_clImageArray);
  }
  if(m_clImageProjections != NULL) {
      clReleaseMemObject(m_clImageProjections);
  }
}

void CphotoSetS::freePhotoSet(const int level) {
  for (int index = 0; index < (int)m_photos.size(); ++index)
    m_photos[index].free(level);
}

void CphotoSetS::setEdge(const float threshold) {
  for (int index = 0; index < m_num; ++index)
    m_photos[index].setEdge(threshold);
}

void CphotoSetS::write(const std::string outdir) {
  for (int index = 0; index < m_num; ++index) {
    const int image = m_images[index];
    char buffer[1024];
    sprintf(buffer, "%s%08d.txt", outdir.c_str(), image);
    
    m_photos[index].write(buffer);
  }
}

// get x and y axis to collect textures given reference index and normal
void CphotoSetS::getPAxes(const int index, const Vec4f& coord, const Vec4f& normal,
                          Vec4f& pxaxis, Vec4f& pyaxis) const{
  m_photos[index].getPAxes(coord, normal, pxaxis, pyaxis);
}

void CphotoSetS::grabTex(const int index, const int level, const Vec2f& icoord,
                         const Vec2f& xaxis, const Vec2f& yaxis,
                         std::vector<Vec3f>& tex, const int normalizef) const{
  m_photos[index].grabTex(level, icoord, xaxis, yaxis, m_size, tex, normalizef);
}

// grabTex given 3D sampling information
void CphotoSetS::grabTex(const int index, const int level, const Vec4f& coord,
                         const Vec4f& pxaxis, const Vec4f& pyaxis, const Vec4f& pzaxis,
                         std::vector<Vec3f>& tex, float& weight,
                         const int normalizef) const {
  m_photos[index].grabTex(level, coord, pxaxis, pyaxis, pzaxis,
                          m_size, tex, weight, normalizef);
}

float CphotoSetS::incc(const std::vector<std::vector<Vec3f> >& texs,
                       const std::vector<float>& weights) {
  float incctmp = 0.0;
  float denom = 0.0;
  for (int i = 0; i < (int)weights.size(); ++i) {
    if (texs[i].empty())
      continue;
    for (int j = i+1; j < (int)weights.size(); ++j) {
      if (texs[j].empty())
	continue;
      
      const float weight = weights[i] * weights[j];
      const float ftmp = Cphoto::idot(texs[i], texs[j]);
      incctmp += ftmp * weight;
      denom += weight;
    }
  }
  
  if (denom == 0.0)
    return 2.0f;
  else
    return incctmp / denom;
}

void CphotoSetS::getMinMaxAngles(const Vec4f& coord, const std::vector<int>& indexes,
                                 float& minAngle, float& maxAngle) const {
  minAngle = M_PI;
  maxAngle = 0.0f;
  vector<Vec4f> rays;  rays.resize((int)indexes.size());
  for (int i = 0; i < (int)indexes.size(); ++i) {
    const int index = indexes[i];
    rays[i] = m_photos[index].m_center - coord;
    unitize(rays[i]);
  }
  
  for (int i = 0; i < (int)indexes.size(); ++i) {
    for (int j = i+1; j < (int)indexes.size(); ++j) {
      const float dot = max(-1.0f, min(1.0f, rays[i] * rays[j]));
      const float angle = acos(dot);
      minAngle = min(angle, minAngle);
      maxAngle = max(angle, maxAngle);
    }
  }
}

int CphotoSetS::checkAngles(const Vec4f& coord,
                            const std::vector<int>& indexes,
                            const float minAngle, const float maxAngle,
                            const int num) const {
  int count = 0;
  
  vector<Vec4f> rays;  rays.resize((int)indexes.size());
  for (int i = 0; i < (int)indexes.size(); ++i) {
    const int index = indexes[i];
    rays[i] = m_photos[index].m_center - coord;
    unitize(rays[i]);
  }
  
  for (int i = 0; i < (int)indexes.size(); ++i) {
    for (int j = i+1; j < (int)indexes.size(); ++j) {
      const float dot = max(-1.0f, min(1.0f, rays[i] * rays[j]));
      const float angle = acos(dot);
      if (minAngle < angle && angle < maxAngle)
        ++count;
    }
  }

  //if (count < num * (num - 1) / 2)
  if (count < 1)
    return 1;
  else
    return 0;
}

float CphotoSetS::computeDepth(const int index, const Vec4f& coord) const {
  return m_photos[index].computeDepth(coord);
}

void CphotoSetS::setDistances(void) {
  m_distances.resize(m_num);
  float avedis = 0.0f;
  int denom = 0;
  for (int i = 0; i < m_num; ++i) {
    m_distances[i].resize(m_num);
    for (int j = 0; j < m_num; ++j) {
      if (i == j)
        m_distances[i][j] = 0.0f;
      else {
        const float ftmp = norm(m_photos[i].m_center - m_photos[j].m_center);
        m_distances[i][j] = ftmp;
        avedis += ftmp;
        denom++;
      }
    }
  }
  if (denom == 0)
    return;
  
  avedis /= denom;
  if (avedis == 0.0f) {
    cerr << "All the optical centers are identical..?" << endl;
    exit (1);
  }
  
  // plus angle difference
  for (int i = 0; i < m_num; ++i) {
    Vec4f ray0 = m_photos[i].m_oaxis;
    ray0[3] = 0.0f;
    for (int j = 0; j < m_num; ++j) {
      Vec4f ray1 = m_photos[j].m_oaxis;
      ray1[3] = 0.0f;
      
      m_distances[i][j] /= avedis;
      const float margin = cos(10.0f * M_PI / 180.0f);
      const float dis = max(0.0f, 1.0f - ray0 * ray1 - margin);
      m_distances[i][j] += dis;
    }
  }
}

int CphotoSetS::image2index(const int image) const {
  map<int, int>::const_iterator pos = m_dict.find(image);
  if (pos == m_dict.end())
    return -1;
  else
    return pos->second;
}

void CphotoSetS::rgbToRGBA(int width, int height, unsigned char *in, unsigned char *out) {
    unsigned char *cin = in;
    unsigned char *cout = out;
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++) {
            for(int k=0; k<3; k++) {
                *cout = *cin;
                cin++;
                cout++;
            }
            *cout = 255;
            cout++;
        }
    }
}
