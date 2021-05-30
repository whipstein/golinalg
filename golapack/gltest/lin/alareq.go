package lin

// Alareq handles input for the LAPACK test program.  It is called
// to evaluate the input line which requested NMATS matrix types for
// PATH.  The flow of control is as follows:
//
// If NMATS = NTYPES then
//    DOTYPE(1:NTYPES) = .TRUE.
// else
//    Read the next input line for NMATS matrix types
//    Set DOTYPE(I) = .TRUE. for each valid type I
// endif
func Alareq(nmats *int, dotype *[]bool) {
	for i := 1; i <= (*nmats); i++ {
		(*dotype)[i-1] = true
	}
}
