#include <stdio.h>
#include <mpi.h>
#include <stddef.h>

#define square(x) (x * x)

typedef struct Workload {
    double a;
    double b;
    int n;
} workload_t;

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype type;
    const int block_lengths[3] = {1, 1, 1};
    const MPI_Aint displacements[3] = {offsetof(workload_t, a), offsetof(workload_t, b), offsetof(workload_t, n)};;
    const MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    MPI_Type_create_struct(3, block_lengths, displacements, types, &type);
    MPI_Type_commit(&type);
    workload_t workload;
    if (rank == 0) {
        int n;
        printf("Enter a, b, n\n");
        scanf("%lf %lf %d", &workload.a, &workload.b, &workload.n);
        // send data
        MPI_Bcast(&workload, 1, type, 0, MPI_COMM_WORLD);
    }
    // receive data
    MPI_Bcast(&workload, 1, type, 0, MPI_COMM_WORLD);

    double step_size = (workload.b - workload.a) / workload.n;
    int local_n = workload.n / size;
    double local_a = workload.a + rank * local_n * step_size;
    double local_b = local_a + step_size * local_n;
    double sum = 0;
    for (int i = 1; i < local_n - 1; ++i) {
        double x = local_a + i * step_size;
        sum += square(x);
    }
    sum = step_size * (0.5 * square(local_a) + 0.5 * square(local_b) + sum);
    double result = 0;
    MPI_Reduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("The result of the integral is %lf\n", result);
    }
    MPI_Finalize();
    return 0;
}
