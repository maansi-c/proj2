/*
 * This file is part of libFirm.
 * Copyright (C) 2018 Christoph Mallon.
 */

#ifndef FIRM_BE_RISCV_RISCV_BEARCH_T_H
#define FIRM_BE_RISCV_RISCV_BEARCH_T_H

#define RISCV_PO2_STACK_ALIGNMENT 4
#define RISCV_N_PARAM_REGS  8

#include <stdbool.h>
#include <stdint.h>

#include "beirg.h"
#include "firm_types.h"

extern unsigned riscv_machine_size;
extern unsigned riscv_register_size;
extern unsigned riscv_param_stack_align;

extern ir_mode *mode_rv_gp;

typedef struct riscv_irg_data_t {
	bool     omit_fp;        /**< No frame pointer is used. */
} riscv_irg_data_t;

static inline riscv_irg_data_t *riscv_get_irg_data(const ir_graph *irg)
{
	return (riscv_irg_data_t*)be_birg_from_irg(irg)->isa_link;
}

static inline bool riscv_is_32(void)
{
	assert(riscv_machine_size == 32 || riscv_machine_size == 64);
	return riscv_machine_size == 32;
}

static inline bool riscv_is_64(void)
{
	assert(riscv_machine_size == 32 || riscv_machine_size == 64);
	return riscv_machine_size == 64;
}

static inline bool is_uimm5_6(long const val)
{
	if (riscv_is_32()) {
		return 0 <= val && val < 32;
	} else {
		return 0 <= val && val < 64;
	}
}

static inline bool is_simm12(long const val)
{
	return -2048 <= (int32_t)val && (int32_t)val < 2048;
}

static inline bool tarval_is_simm12(ir_tarval const *val)
{
	// For modes up to 32 bits, we can use the faster and mode-agnostic is_simm12.
	unsigned bits = get_mode_size_bits(get_tarval_mode(val));

	if (bits <= 32) {
		return is_simm12(get_tarval_long(val));
	} else if (bits == 64) {
		assert(riscv_is_64() && "unlowered 64-bit value");

		ir_tarval *left_shifted = tarval_shl_unsigned(val, 64 - 12);
		ir_tarval *right_shifted = tarval_shrs_unsigned(left_shifted, 64 - 12);

		return tarval_cmp(val, right_shifted) == ir_relation_equal;
	}

	panic("Unexpected mode");
}

typedef struct riscv_hi_lo_imm {
	int32_t hi;
	int32_t lo;
} riscv_hi_lo_imm;

riscv_hi_lo_imm calc_hi_lo(int32_t val);

void riscv_set_xlen(unsigned xlen);

void riscv_finish_graph(ir_graph *irg);

#endif
