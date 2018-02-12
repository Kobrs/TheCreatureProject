#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _dop_expsyn_reg(void);
extern void _dop_ExpSynSTDP_reg(void);
extern void _stdp_reg(void);
extern void _vecstim_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," dop_expsyn.mod");
    fprintf(stderr," dop_ExpSynSTDP.mod");
    fprintf(stderr," stdp.mod");
    fprintf(stderr," vecstim.mod");
    fprintf(stderr, "\n");
  }
  _dop_expsyn_reg();
  _dop_ExpSynSTDP_reg();
  _stdp_reg();
  _vecstim_reg();
}
