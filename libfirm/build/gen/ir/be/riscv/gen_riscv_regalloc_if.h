/**
 * @file
 * @brief Contains additional external requirements defs for external includes.
 * @note   DO NOT EDIT THIS FILE, your changes will be lost.
 *         Edit ./ir/be/riscv/riscv_spec.pl instead.
 *         created by: ./ir/be/scripts/generate_regalloc_if.pl ./ir/be/riscv/riscv_spec.pl build/gen/ir/be/riscv
 * @date   Wed Sep 27 23:59:06 2023
 */
#ifndef FIRM_BE_RISCV_GEN_RISCV_REGALLOC_IF_H
#define FIRM_BE_RISCV_GEN_RISCV_REGALLOC_IF_H

#include "bearch.h"

/** global register indices for riscv registers */
enum reg_indices {
	REG_S0,
	REG_S1,
	REG_A0,
	REG_A1,
	REG_A2,
	REG_A3,
	REG_A4,
	REG_A5,
	REG_ZERO,
	REG_RA,
	REG_SP,
	REG_GP,
	REG_TP,
	REG_T0,
	REG_T1,
	REG_T2,
	REG_A6,
	REG_A7,
	REG_S2,
	REG_S3,
	REG_S4,
	REG_S5,
	REG_S6,
	REG_S7,
	REG_S8,
	REG_S9,
	REG_S10,
	REG_S11,
	REG_T3,
	REG_T4,
	REG_T5,
	REG_T6,

	N_RISCV_REGISTERS
};

/** local register indices for riscv registers */
enum {
	REG_GP_S0,
	REG_GP_S1,
	REG_GP_A0,
	REG_GP_A1,
	REG_GP_A2,
	REG_GP_A3,
	REG_GP_A4,
	REG_GP_A5,
	REG_GP_ZERO,
	REG_GP_RA,
	REG_GP_SP,
	REG_GP_GP,
	REG_GP_TP,
	REG_GP_T0,
	REG_GP_T1,
	REG_GP_T2,
	REG_GP_A6,
	REG_GP_A7,
	REG_GP_S2,
	REG_GP_S3,
	REG_GP_S4,
	REG_GP_S5,
	REG_GP_S6,
	REG_GP_S7,
	REG_GP_S8,
	REG_GP_S9,
	REG_GP_S10,
	REG_GP_S11,
	REG_GP_T3,
	REG_GP_T4,
	REG_GP_T5,
	REG_GP_T6,
};


/** number of registers in riscv register classes. */
enum {
	N_riscv_gp_REGS = 32,

};

enum {
	CLASS_riscv_gp,
	N_RISCV_CLASSES = 1
};

extern const arch_register_req_t riscv_class_reg_req_gp;
extern const arch_register_req_t riscv_single_reg_req_gp_s0;
extern const arch_register_req_t riscv_single_reg_req_gp_s1;
extern const arch_register_req_t riscv_single_reg_req_gp_a0;
extern const arch_register_req_t riscv_single_reg_req_gp_a1;
extern const arch_register_req_t riscv_single_reg_req_gp_a2;
extern const arch_register_req_t riscv_single_reg_req_gp_a3;
extern const arch_register_req_t riscv_single_reg_req_gp_a4;
extern const arch_register_req_t riscv_single_reg_req_gp_a5;
extern const arch_register_req_t riscv_single_reg_req_gp_zero;
extern const arch_register_req_t riscv_single_reg_req_gp_ra;
extern const arch_register_req_t riscv_single_reg_req_gp_sp;
extern const arch_register_req_t riscv_single_reg_req_gp_gp;
extern const arch_register_req_t riscv_single_reg_req_gp_tp;
extern const arch_register_req_t riscv_single_reg_req_gp_t0;
extern const arch_register_req_t riscv_single_reg_req_gp_t1;
extern const arch_register_req_t riscv_single_reg_req_gp_t2;
extern const arch_register_req_t riscv_single_reg_req_gp_a6;
extern const arch_register_req_t riscv_single_reg_req_gp_a7;
extern const arch_register_req_t riscv_single_reg_req_gp_s2;
extern const arch_register_req_t riscv_single_reg_req_gp_s3;
extern const arch_register_req_t riscv_single_reg_req_gp_s4;
extern const arch_register_req_t riscv_single_reg_req_gp_s5;
extern const arch_register_req_t riscv_single_reg_req_gp_s6;
extern const arch_register_req_t riscv_single_reg_req_gp_s7;
extern const arch_register_req_t riscv_single_reg_req_gp_s8;
extern const arch_register_req_t riscv_single_reg_req_gp_s9;
extern const arch_register_req_t riscv_single_reg_req_gp_s10;
extern const arch_register_req_t riscv_single_reg_req_gp_s11;
extern const arch_register_req_t riscv_single_reg_req_gp_t3;
extern const arch_register_req_t riscv_single_reg_req_gp_t4;
extern const arch_register_req_t riscv_single_reg_req_gp_t5;
extern const arch_register_req_t riscv_single_reg_req_gp_t6;


extern const arch_register_t riscv_registers[N_RISCV_REGISTERS];

extern arch_register_class_t riscv_reg_classes[N_RISCV_CLASSES];

void riscv_register_init(void);

#endif
