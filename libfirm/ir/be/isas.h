/*
 * This file is part of libFirm.
 * Copyright (C) 2017 University of Karlsruhe.
 */
#ifndef FIRM_BE_ISAS_H
#define FIRM_BE_ISAS_H

#include "firm_types.h"
#include "bearch.h"

void be_init_arch_riscv32(void);
extern arch_isa_if_t const riscv32_isa_if;

#endif
