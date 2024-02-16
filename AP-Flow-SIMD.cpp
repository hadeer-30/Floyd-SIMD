/*

Submitted by: Hadeer Farahat

Submission for FA21, COSC 594 - lab5 

This file includes the implementation of CalcFlow() function from APFlow class.
The main goal of this function is to calculate the maximum flow path between each pair of nodes.
This function uses Floyd-Warshall algorithm using SIMD instructions to exploit instruction-level parallelism,
and speedup the overall running time.

*/

#include "AP-Flow.h"
#include <emmintrin.h>
#include <immintrin.h>
using namespace std;

void APFlow::CalcFlow(){
  int i, j, v;
  __m128i alli,fv; 
  //__m128i rv; // not needed , can use the iv pointer instead as it already points to the same region 
  __m128i *vv,*iv;

  // Initializing Flow by the Adjacency matrix
  for (i = 0; i < N*N; i++) Flow[i] = Adj[i];

  for (v = 0; v < N; v++) {    
    for (i = 0; i < N; i++) {  
      /* Create a vector alli, which is 16 instances of Flow[i*N+v] */   
      alli = _mm_set1_epi8(Flow[i*N+v]);

      //rv = (__m128i *)(Flow+(i*N)); // rv is no longer needed 

      /* Load Flow[i*N+j] through Flow[i*N+j+15] to vector iv.
         now, iv is pointing to Flow[i*N].
         In the J-loop, it will handle each 16 element
      */
      iv = (__m128i *)(Flow+(i*N));

      /* Load Flow[v*N+j] through Flow[v*N+j+15] to vector vv.
         now, vv is pointing to Flow[v*N].
         In the J-loop, it will handle each 16 element
      */      
      vv = (__m128i *)(Flow+(v*N));
      for (j = 0; j < N; j += 16) {
        
        //iv = _mm_loadu_si128((__m128i*)(Flow+(i*N+j))); // used pointer instead to fix timing issue
        //vv = _mm_loadu_si128((__m128i*)(Flow+(v*N+j))); 

        /* Create fv, which is the flow from i to each of j through j+15
           through v. This is simply the min of alli and vv. */
        fv = _mm_min_epu8(alli,*vv);

        /* Create rv, which is the max of each value of fv and iv. */
        /* Store rv into Flow[i*N+j] through Flow[i*N+j+15] */

        //the next line substitutes the creation and storing to rv             
        *iv = _mm_max_epu8(fv,*iv); // no need to rv

        /* 
        rv = = _mm_max_epu8(fv,*iv);
        *iv=rv; // iv is a pointer to (__m128i*)(Flow+i*N+j) , it works but is not needed (iv can handle it in the previous line)
        */
        //_mm_storeu_si128((__m128i*)(Flow+i*N+j),rv); //correct, but it is time consuming

        //increament the pointers to handle the next 16 elements
        iv++;
        vv++;
      }   
    }
  }
}