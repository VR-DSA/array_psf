// -*- c++ -*-
/*

Input text file with space-separated x and y offset positions.
Define some parameters in code on number of frequency channels. 
TODO: include time

 */
#include <iostream>
#include <algorithm>
using std::cout;
using std::cerr;
using std::endl;
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/syscall.h>
#include <syslog.h>
#include <curand.h>
#include <curand_kernel.h>

#define CVAC 2.998e8
#define NANT 2016
#define NBASEL NANT*(NANT-1)/2
#define NCHAN 1
#define FCH1_MHZ 1350.0
#define CHBW_MHZ 64.0
#define NTHREADS_GPU 32
#define NSIDE 1024
#define PIX_ASEC 0.5
#define N_INTENSITY 8

// kernel to calculate uv coordinates from antenna positions
// run with NBASEL*NCH/NTHREADS_GPU/N_INTENSITY blocks of NTHREADS_GPU threads
__global__
void calc_uv(int *a1, int *a2, float *ant_x, float *ant_y, float *freqs, float *u, float *v) {

  int idx = N_INTENSITY*(blockIdx.x*NTHREADS_GPU + threadIdx.x);
  int bidx,chidx;
  
  for (int i=idx;i<idx+N_INTENSITY;i++) {
  
    bidx = (int)(i / NCHAN);
    chidx = (int)(i % NCHAN);

    u[i] = (freqs[chidx]/CVAC)*(ant_x[a2[bidx]]-ant_x[a1[bidx]]);
    v[i] = (freqs[chidx]/CVAC)*(ant_y[a2[bidx]]-ant_y[a1[bidx]]);
  }
    
}

// kernel to calculate PSF
// needs to integrate over UVs to get image. 
// run with NSIDE*NSIDE/NTHREADS_GPU blocks of NTHREADS_GPU threads
__global__
void calc_psf(float *u, float *v, float *psf) {

  int idx = blockIdx.x*NTHREADS_GPU + threadIdx.x;
  int li = (int)(idx/NSIDE);
  int mi = (int)(idx % NSIDE);

  float l = (li*1.-NSIDE/2.)*((PIX_ASEC/3600.)*M_PI/180.);
  float m = (mi*1.-NSIDE/2.)*((PIX_ASEC/3600.)*M_PI/180.);

  for (int i=0;i<NBASEL*NCHAN;i++) {
    if (u[i]*u[i] + v[i]*v[i] > 900.) 
      psf[idx] += cosf(2.*M_PI*(u[i]*l+v[i]*m));
  }
  
}

// function to read in antenna positions
void read_ants(char * fnam, float *ant_x, float *ant_y) {

  FILE *fin;
  fin=fopen(fnam,"r");
  for (int i=0;i<NANT;i++)
    fscanf(fin,"%f %f\n",&ant_x[i],&ant_y[i]);
  fclose(fin);

}

// function to init freqs
void init_freqs(float * freqs) {

  for (int i=0;i<NCHAN;i++)
    freqs[i] = FCH1_MHZ*1e6+CHBW_MHZ*1e6*i;

}

// function to fill a1 and a2 arrays
void fill_aas(int *a1, int *a2) {

  int ii = 0;
  for (int i=0;i<NANT-1;i++) {
    for (int j=i+1;j<NANT;j++) {
      a1[ii] = i;
      a2[ii] = j;
      ii++;
    }
  }
}

void usage() {

  printf("gen_psf -f <antenna file name>\n");
  
}

int main(int argc, char **argv) {

  // command line arguments
  char * fnam = (char *)malloc(sizeof(char)*200);
  for (int i=1;i<argc;i++) {
    if (strcmp(argv[i],"-f")==0) {
      strcpy(fnam,argv[i+1]);
    }
    if (strcmp(argv[i],"-h")==0) {
      usage();
      exit(1);
    }
  }

  printf("Read all command line args\n");
  printf("NBASEL is %d\n",NBASEL);

  // define all host arrays
  float * ant_x = (float *)malloc(sizeof(float)*NANT);
  float * ant_y = (float *)malloc(sizeof(float)*NANT);
  int * a1  = (int *)malloc(sizeof(int)*NBASEL);
  int * a2  = (int *)malloc(sizeof(int)*NBASEL);
  float * freqs = (float *)malloc(sizeof(float)*NCHAN);
  float * psf = (float *)malloc(sizeof(float)*NSIDE*NSIDE);
  float * h_u = (float *)malloc(sizeof(float)*NCHAN*NBASEL);
  float * h_v = (float *)malloc(sizeof(float)*NCHAN*NBASEL);
  
  // define all device arrays
  float *d_ant_x, *d_ant_y, *d_freqs, *u, *v, *d_psf;
  int *d_a1, *d_a2;
  cudaMalloc((void **)&d_ant_x, NANT*sizeof(float));
  cudaMalloc((void **)&d_ant_y, NANT*sizeof(float));
  cudaMalloc((void **)&d_freqs, NCHAN*sizeof(float));
  cudaMalloc((void **)&u, NBASEL*NCHAN*sizeof(float));
  cudaMalloc((void **)&v, NBASEL*NCHAN*sizeof(float));
  cudaMalloc((void **)&d_psf, NSIDE*NSIDE*sizeof(float));
  cudaMalloc((void **)&d_a1, NBASEL*sizeof(int));
  cudaMalloc((void **)&d_a2, NBASEL*sizeof(int));

  // init ant pos
  read_ants(fnam,ant_x,ant_y);
  printf("Read ant pos\n");

  // init freqs
  init_freqs(freqs);
  printf("init_freqs\n");

  // init a1/a2
  fill_aas(a1,a2);
  printf("Fill a1 a2\n");

  // copy to device
  cudaMemcpy(d_ant_x, ant_x, NANT*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ant_y, ant_y, NANT*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_freqs, freqs, NCHAN*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a1, a1, NBASEL*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a2, a2, NBASEL*sizeof(int), cudaMemcpyHostToDevice);
  printf("Copied to device\n");

  // run kernels
  calc_uv<<<NBASEL*NCHAN/NTHREADS_GPU/N_INTENSITY,NTHREADS_GPU>>>(d_a1, d_a2, d_ant_x, d_ant_y, d_freqs, u, v);
  cudaDeviceSynchronize();
  printf("Calc uv\n");
  calc_psf<<<NSIDE*NSIDE/NTHREADS_GPU,NTHREADS_GPU>>>(u,v,d_psf);
  cudaDeviceSynchronize();
  printf("Calc PSF\n");

  // copy to host
  cudaMemcpy(psf, d_psf, NSIDE*NSIDE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_u, u, NBASEL*NCHAN*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_v, v, NBASEL*NCHAN*sizeof(float), cudaMemcpyDeviceToHost);
  printf("copied to host\n");
  
  // write to file
  FILE *fout;
  fout=fopen("tmp.dat","w");
  for (int i=0;i<NSIDE*NSIDE;i++) fprintf(fout,"%f\n",psf[i]);
  fclose(fout);
  fout=fopen("tmp_uv.dat","w");
  for (int i=0;i<NCHAN*NBASEL;i++) fprintf(fout,"%f %f\n",h_u[i],h_v[i]);
  fclose(fout);

  printf("writte to tmp.dat\n");
  
  free(ant_x);
  free(ant_y);
  free(a1);
  free(a2);
  free(psf);
  free(freqs);
  free(h_u);
  free(h_v);
  cudaFree(d_ant_x);
  cudaFree(d_ant_y);
  cudaFree(d_freqs);
  cudaFree(u);
  cudaFree(v);
  cudaFree(d_psf);
  cudaFree(d_a1);
  cudaFree(d_a2);
  
  free(fnam);
  

}
