//
// Created by francesco on 11/14/24.
//

#ifndef COMMONS_H
#define COMMONS_H

#define rk_printf(rank, ...)\
    do {\
        int _rk_printf_rank;\
        MPI_Comm_rank(MPI_COMM_WORLD, &_rk_printf_rank);\
        if (rank == _rk_printf_rank)\
            printf(__VA_ARGS__);\
    } while (0)\

#endif //COMMONS_H
