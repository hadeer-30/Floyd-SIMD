/*

Submitted by: Hadeer Farahat

Submission for FA21 COSC 594 - lab5 

This file includes the implementation of solve() function from TheTips class.
The function solves The-Tips problem from Topcoder.
The goal of this function is claculating how many `eggs` could be found using some clues and probabilities.
This problem is solved by Floyd-Warshall algorithm, and some SIMD instructions are applied-
to exploit an instruction-level parallelism to speed-up the runtime and acheive a better performance.

*/

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <emmintrin.h>
#include <The-Tips.h>

using namespace std;

//packing the matrix into bits, and then using _mm_or_si128()
double TheTips::solve(vector <string> clues, vector <int> probability, int print){
    int i, j, v;
    int N = clues.size();   //retreiving the size of the clues vector to create the matrix
    int rowSize = N*sizeof(char);   // calculating the number of bytes for each row for allocation

    //Pad the number of row bytes so that it's a multiple of 16 (the simd instruction will work on a chunck of 16 elements together)
    if (rowSize % 16 != 0) rowSize += (16 - rowSize%16);

    //create the matrix with prober allignment. Dealing with matrix as 1D vector for easier access with simd instructions.
    __attribute__ ((aligned(16))) char *c = (char *) malloc(rowSize*N);
    
    vector <double> p;
    double x;
        
    /* Change the Y/N's to 1/0's */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            c[i*rowSize+j] = (clues[i][j] == 'Y') ? 1 : 0;  // initializing the matrix
        }
        for(j=N;j<rowSize;j++) c[i*rowSize+j] = 0 ; // initializing the padded bytes to 0
        c[i*rowSize+i] = 1; // c[i][i] = 1 (the node with itself)
    }
    
    /* Print the adjacency matrix before doing Floyd-Warshall */
    if (print) {
        printf("The Adjacency Matrix:\n\n");
        printf("    ");
        for (i = 0; i < N; i++) printf("%X", i&0xf);
        printf("\n");
        printf("   -");
        for (i = 0; i < N; i++) printf("-");
        printf("\n");
        for (i = 0; i < N; i++) {
        printf("%X | ", i&0xf);
        for (j = 0; j < N; j++) printf("%d", c[i*rowSize+j]);
        printf("\n");
        }
        printf("\n");
    }

    
    //__m128i *a,*b,*r;
    __m128i a,b,r;

    for (v = 0; v < N; v++) {
        for (i = 0; i < N; i++) {
            if (c[i*rowSize+v]) {
                for (j = 0; j < N; j+=16) {
                    //use SIMD instructions

                    //loading the proper region from matrix and deal with it as a 128 bits vector (16 elements at a time)
                    a = _mm_loadu_si128((__m128i*)(c+(i*rowSize+j)));
                    r = _mm_loadu_si128((__m128i*)(c+(i*rowSize+j)));
                    b = _mm_loadu_si128((__m128i*)(c+(v*rowSize+j))); 

                    // r = a OR b
                    r = _mm_or_si128(a,b);
                    
                    //store the result from r to the matrix memory
                    _mm_storeu_si128((__m128i*)(c+(i*rowSize+j)),r); 
                }
            }
        }
    }
    

    if (print) {
        printf("Probabilities:\n\n");
        for (i = 0; i < probability.size(); i++) printf("%4d", probability[i]);
        printf("\n\n");
        printf("The Floyd-Warshall Matrix:\n\n");
    
        printf("    ");
        for (i = 0; i < N; i++) printf("%X", i&0xf);
        printf("\n");
        printf("   -");
        for (i = 0; i < N; i++) printf("-");
        printf("\n");
        for (i = 0; i < N; i++) {
        printf("%X | ", i&0xf);
        for (j = 0; j < N; j++) printf("%d", c[i*rowSize+j]);
        printf("\n");
        }
        printf("\n");
    }

    /* Calculate the values of p from the probabilities and reachability: */
    p.resize(N, 0);

    for (i = 0; i < N; i++) {
        x = probability[i];
        x /= 100.0;
        for (j = 0; j < N; j++) {
            if (c[i*rowSize+j]) p[j] += ((1 - p[j]) * x);
        }
    }

    if (print) {
        printf("\nThe Expected Values\n\n");
        for (i = 0; i < N; i++) {
        printf("I: %X    Prob: %7.5lf\n", i, p[i]);
        }
        printf("\n");
    }
        
    /* Calculate the final return value */
    x = 0;
    for (i = 0; i < N; i++) x += p[i];
    
    return x;
}