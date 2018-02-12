#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _iaf_I_reg(void);
extern void _ingauss_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," iaf_I.mod");
    fprintf(stderr," ingauss.mod");
    fprintf(stderr, "\n");
  }
  _iaf_I_reg();
  _ingauss_reg();
}
