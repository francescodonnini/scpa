#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4
#define N 4
#define free_if_true(ptr, v) do { if ((v)) free(ptr); } while(0)

int get_data(double **buffer, int m, int n);
void get_send_counts_and_displs(int *send_counts, int *displs, int rows, int size);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *mtx;
    double *vec;
    int *sendcounts;
    int *displs;
    if (rank == 0) {
        if (get_data(&mtx, M, N)) {
            goto end;
        }
        if (get_data(&vec, M, 1)) {
            free(mtx);
            goto end;
        }
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%lf, ", mtx[i * N + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("\n");
        for (int j = 0; j < N; ++j) {
            printf("%lf, ", mtx[j]);
        }
        printf("\n");
    }
    sendcounts = malloc(size * sizeof(int));
    if (sendcounts == NULL) {
        free_if_true(mtx, rank == 0);
        free_if_true(vec, rank == 0);
        goto end;
    }
    displs = malloc(size * sizeof(int));
    if (displs == NULL) {
        free_if_true(mtx, rank == 0);
        free_if_true(vec, rank == 0);
        free(sendcounts);
        goto end;
    }
    get_send_counts_and_displs(sendcounts, displs, M, size);
    double *mtx_buffer;
    if (rank != 0) {
        mtx_buffer = malloc(sendcounts[rank]*sizeof(double));
        if (mtx_buffer == NULL) {
            free_if_true(mtx, rank == 0);
            free_if_true(vec, rank == 0);
            free(sendcounts);
            free(displs);
            goto end;
        }
        MPI_Scatterv(mtx, sendcounts, displs, MPI_DOUBLE, mtx_buffer, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(mtx, sendcounts, displs, MPI_DOUBLE, mtx, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    double *vec_buffer = malloc(N * sizeof(double));
    if (vec_buffer == NULL) {
        free_if_true(mtx, rank == 0);
        free_if_true(vec, rank == 0);
        free(sendcounts);
        free(displs);
        free_if_true(mtx_buffer, rank != 0);
        goto end;
    }
    if (rank == 0) {
        MPI_Bcast(vec, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(vec_buffer, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double *sums = malloc(sizeof(double) * sendcounts[rank]);
    for (int i = 0; i < sendcounts[rank]; ++i) {
        sums[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            if (rank == 0) {
                sums[i] += mtx[i * sendcounts[rank] + j] * vec[j];
            } else {
                sums[i] += mtx_buffer[i * sendcounts[rank] + j] * vec_buffer[j];
            }
        }
    }

    MPI_Gatherv(sums, sendcounts[rank], MPI_DOUBLE, vec, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("the result:\n");
        for (int i = 0; i < N; ++i) {
            printf("%lf,", vec[i]);
        }
        printf("\n");
    }

    free_if_true(mtx, rank == 0);
    free_if_true(vec, rank == 0);
    free(sendcounts);
    free(displs);
    free_if_true(mtx_buffer, rank != 0);
    free(vec_buffer);
end:
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

void get_send_counts_and_displs(int *send_counts, int *displs, int rows, int size) {
    int chunk = rows / size;
    int *sp = send_counts;
    int *dp = displs;
    for (int count = 0; count <= rows; count += chunk) {
        *sp++ = chunk;
        *dp++ = count;
    }
    if (rows % size != 0) {
        *(sp-1) += rows % size;
    }
}