package golapack

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
func Alareq(path []byte, nmats *int, dotype *[]bool, ntypes *int) {
	// var firstt bool
	// var c1 byte
	// var i, i1, ic, j, k, lenp, nt int
	var i int
	// intstr := make([]byte, 10)
	// line := make([]byte, 80)
	// nreq := make([]int, 100)

	// intstr[0], intstr[1], intstr[2], intstr[3], intstr[4], intstr[5], intstr[6], intstr[7], intstr[8], intstr[9] = '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'

	// if (*nmats) >= (*ntypes) {
	//        Test everything if NMATS >= NTYPES.
	for i = 1; i <= (*nmats); i++ {
		(*dotype)[i-1] = true
	}
	// } else {
	// 	for i = 1; i <= (*ntypes); i++ {
	// 		(*dotype)[i-1] = false
	// 	}
	// }
	// firstt = true

	// 	//        Read a line of matrix types if 0 < NMATS < NTYPES.
	// 	if (*nmats) > 0 {
	// 		intrinsic.READ(nin, []byte("%80s\n"), LINE)
	// 		lenp = len(line)
	// 		i = 0
	// 		for j = 1; j <= (*nmats); j++ {
	// 			nreq[j-1] = 0
	// 			i1 = 0
	// 		label30:
	// 			;
	// 			i = i + 1
	// 			if i > lenp {
	// 				if j == (*nmats) && i1 > 0 {
	// 					goto label60
	// 				} else {
	// 					intrinsic.WRITE(*nout, func() *[]byte { y := []byte("%v *** Not enough matrix types on input line\n%79s\n"); return &y }(), line)
	// 					intrinsic.WRITE(*nout, func() *[]byte {
	// 						y := []byte(" ==> Specify %4d matrix types on this line or adjust NTYPES on previous line\n")
	// 						return &y
	// 					}(), *nmats)
	// 					goto label80
	// 				}
	// 			}
	// 			if line[i-1] != ' ' && line[i-1] != ',' {
	// 				i1 = i
	// 				c1 = line[i1-1]
	// 				//
	// 				//              Check that a valid integer was read
	// 				//
	// 				for k = 1; k <= 10; k++ {
	// 					if c1 == intstr[k-1] {
	// 						ic = k - 1
	// 						goto label50
	// 					}
	// 					//Label40:
	// 				}
	// 				intrinsic.WRITE(*nout, func() *[]byte {
	// 					y := []byte("%v *** Invalid integer value in column %2d of input line:\n%79s\n")
	// 					return &y
	// 				}(), i, line)
	// 				intrinsic.WRITE(*nout, func() *[]byte {
	// 					y := []byte(" ==> Specify %4d matrix types on this line or adjust NTYPES on previous line\n")
	// 					return &y
	// 				}(), *nmats)
	// 				goto label80
	// 			label50:
	// 				;
	// 				nreq[j-1] = 10*nreq[j-1] + ic
	// 				goto label30
	// 			} else if i1 > 0 {
	// 				goto label60
	// 			} else {
	// 				goto label30
	// 			}
	// 		label60:
	// 		}
	// 	}
	// 	for i = 1; i <= (*nmats); i++ {
	// 		nt = nreq[i-1]
	// 		if nt > 0 && nt <= (*ntypes) {
	// 			if (*dotype)[nt-1] {
	// 				if firstt {
	// 					intrinsic.WRITE(*nout, func() *[]byte { y := []byte(" %v\n"); return &y }())
	// 				}
	// 				firstt = false
	// 				intrinsic.WRITE(*nout, func() *[]byte {
	// 					y := []byte(" *** Warning:  duplicate request of matrix type %2d for %3s\n")
	// 					return &y
	// 				}(), nt, *path)
	// 			}
	// 			(*dotype)[nt-1] = true
	// 		} else {
	// 			intrinsic.WRITE(*nout, func() *[]byte {
	// 				y := []byte(" *** Invalid type request for %3s, type  %4d: must satisfy  1 <= type <= %2d\n")
	// 				return &y
	// 			}(), *path, nt, *ntypes)
	// 		}
	// 		//Label70:
	// 	}
	// label80:
	// }
	// return
	// //
	// //Label90:

	// intrinsic.WRITE(*nout, func() *[]byte {
	// 	y := []byte("\n *** End of file reached when trying to read matrix types for %3s\n *** Check that you are requesting the right number of types for each path\n\n")
	// 	return &y
	// }(), *path)
	// intrinsic.WRITE(*nout, func() *[]byte { y := []byte(" %v\n"); return &y }())
	// panic("")
	// //
	// //     End of ALAREQ
	// //
}
