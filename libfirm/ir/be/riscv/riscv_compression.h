/*
 * This file is part of libFirm.
 */

/**
 * @file
 * @brief       This file supports compression-aware register allocation for the RISC-V backend.
 * @author      Maximilian Stemmer-Grabow
 */

#ifndef FIRM_BE_RISCV_RISCV_COMPRESSION_H
#define FIRM_BE_RISCV_RISCV_COMPRESSION_H

#include <stdbool.h>
#include "firm_types.h"
#include "bera.h"

/**
 * Returns a compression requirement description for the operation @p node.
 *
 * This is used by the copy optimization to decide on register allocation choices
 * to optimize for code compression.
 */
compression_req_t riscv_get_op_compression_requirements(ir_node const *node);

/**
 * Count the number of uses of an operation.
 */
int riscv_get_use_count(const ir_node *node);

/**
 * Check whether an operation will be compressible.
 *
 * This is based on the set registers and is only useful after register allocation
 * is already completed.
 */
bool riscv_is_compressible(const ir_node *node);

#endif
