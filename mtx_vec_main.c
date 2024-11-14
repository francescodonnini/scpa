#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "commons.h"

#define M 9
#define N 9
#define free_if_true(ptr, v) do { if ((v)) free(ptr); } while(0)

int get_data(double **buffer, int m, int n);

void split_from_matrix(int *sendcnts, int *displs, int nrows, int ncols, int size);

void split_from_vector(int *sendcnts, int *displs, int ncols, int size);

void fdot(int nrow, int ncol, double mtx[nrow][ncol], double vec[ncol], double sums[nrow]);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *mtx;
    double *vec;
    if (rank == 0) {
        if (get_data(&mtx, M, N)) {
            printf("cannot get matrix (%d, %d)\n", M, N);
            goto FINALIZE;;
        }
        if (get_data(&vec, N, 1)) {
            printf("cannot get vector (%d, 1)\n", N);
            free(mtx);
            goto FINALIZE;;
        }
        printf("the matrix:\n");
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%lf, ", mtx[i * N + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("the vector:\n");
        for (int j = 0; j < N; ++j) {
            printf("%lf, ", mtx[j]);
        }
        printf("\n");
    }
    int *sendcnts = malloc(sizeof(int) * size);
    if (sendcnts == NULL) {
        printf("cannot make enough space for sendcnts array\n");
        free_if_true(mtx, rank == 0);
        free_if_true(vec, rank == 0);
        goto FINALIZE;
    }
    int *displs = malloc(sizeof(int) * size);
    if (displs == NULL) {
        printf("cannot make enough space for sendcnts array\n");
        free_if_true(mtx, rank == 0);
        free_if_true(vec, rank == 0);
        free(sendcnts);
        goto FINALIZE;
    }
    split_from_matrix(sendcnts, displs, M, N, size);
    double *mtx_buffer;
    if (rank != 0) {
        mtx_buffer = malloc(sendcnts[rank]*sizeof(double));
        if (mtx_buffer == NULL) {
            free(sendcnts);
            free(displs);
            goto FINALIZE;
        }
        MPI_Scatterv(mtx, sendcnts, displs, MPI_DOUBLE, mtx_buffer, sendcnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(mtx, sendcnts, displs, MPI_DOUBLE, mtx, sendcnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    double *vec_buffer;
    if (rank != 0) {
        vec_buffer = malloc(sizeof(double) * N);
        if (vec_buffer == NULL) {
            free(sendcnts);
            free(displs);
            free(mtx_buffer);
            goto FINALIZE;
        }
        MPI_Bcast(vec_buffer, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(vec, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double *sums = malloc(sizeof(double) * sendcnts[rank]);
    // memset((void*)sums, 0, sendcnts[rank] * sizeof(int));
    if (rank == 0) {
        fdot(sendcnts[rank], N, (double(*)[N])mtx, vec, sums);
    } else {
        fdot(sendcnts[rank], N, (double(*)[N])mtx_buffer, vec_buffer, sums);
    }

    split_from_vector(sendcnts, displs, N, size);
    MPI_Gatherv(sums, sendcnts[rank], MPI_DOUBLE, vec, sendcnts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("the result:\n");
        for (int i = 0; i < N; ++i) {
            printf("%lf,", vec[i]);
        }
        printf("\n");
    }

FINALIZE:
    MPI_Finalize();
}

int get_data(double **buffer, int m, int n) {
    double *b = malloc(sizeof(double) * m * n);
    if (b == NULL) {
        return 1;
    }
    srand(42);
    for (int i = 0; i < m * n; i++) {
        b[i] = i;
    }
    *buffer = b;
    return 0;
}

void split_from_matrix(int *sendcnts, int *displs, int nrows, int ncols, int size) {
    int chunk = nrows / size * ncols;
    sendcnts[0] = chunk;
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        sendcnts[i] = chunk;
        displs[i] = displs[i-1] + chunk;
    }
    if (nrows % size != 0) {
        sendcnts[size-1] += (nrows % size) * chunk;
    }
    rk_printf(0, "the send counts:\n");
    for (int i = 0; i < size; ++i) {
        rk_printf(0, "%d, ", sendcnts[i]);
    }
    rk_printf(0, "\n");
    rk_printf(0, "the displacements:\n");
    for (int i = 0; i < size; ++i) {
         rk_printf(0, "%d, ", displs[i]);
    }
}

void split_from_vector(int *sendcnts, int *displs, int ncols, int size) {
    int chunk = ncols / size;
    sendcnts[0] = chunk;
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        sendcnts[i] = chunk;
        displs[i] = displs[i-1] + chunk;
    }
    if (ncols % size != 0) {
        sendcnts[size-1] += ncols % size;
    }
    rk_printf(0, "the send counts:\n");
    for (int i = 0; i < size; ++i) {
        rk_printf(0, "%d, ", sendcnts[i]);
    }
    rk_printf(0, "\n");
    rk_printf(0, "the displacements:\n");
    for (int i = 0; i < size; ++i) {
         rk_printf(0, "%d, ", displs[i]);
    }
}

void fdot(int nrow, int ncol, double mtx[nrow][ncol], double vec[ncol], double sums[nrow]) {
    for (int i = 0; i < nrow; ++i) {
        sums[i] = 0.0;
        for (int j = 0; j < ncol; ++j) {
            sums[i] += mtx[i][j] * vec[j];
        }
    }
}