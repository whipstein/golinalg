package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/mat"
)

// alaerh is an error handler for the LAPACK routines.  It prints the
// header if this is the first error message and prints the error code
// and form of recovery, if any.  The character evaluations in this
// routine may make it slow, but it should not be called once the LAPACK
// routines are fully debugged.
func alaerh(path, subnam string, info, infoe int, opts []byte, m, n, kl, ku, n5, imat, nfail, nerrs int) (nerrsOut int) {
	var uplo mat.MatUplo

	nerrsOut = nerrs

	if info == 0 {
		return
	}
	p2 := path[1:3]
	c3 := subnam[3:]

	if len(c3) > 5 {
		c3 = c3[3:6]
	}

	//     Print the header if this is the first error message.
	if nfail == 0 && nerrsOut == 0 {
		if c3 == "sv" || c3 == "svx" {
			aladhd(path)
		} else {
			alahd(path)
		}
	}
	nerrsOut++

	//     Print the message detailing the error and form of recovery,
	//     if any.
	if p2 == "ge" {
		//        xGE:  General matrices
		if c3 == "trf" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> m=%5d, n=%5d, nb=%4d, type %2d\n", subnam, info, infoe, m, n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s=%5d for m=%5d, n=%5d, nb=%4d, type %2d\n", subnam, info, m, n, n5, imat)
			}
			if info != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if c3 == "sv" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for n=%5d, nrhs=%4d, type %2d\n", subnam, info, n, n5, imat)
			}

		} else if c3 == "svx" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> fact='%c', trans='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, opts[0], opts[1], n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> fact='%c', trans='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], opts[1], n, n5, imat)
			}

		} else if c3 == "tri" {
			fmt.Printf(" *** Error code from %s=%5d for n=%5d, nb=%4d, type %2d\n", subnam, info, n, n5, imat)

		} else if string(subnam[1:6]) == "latms" {
			fmt.Printf(" *** Error code from %s =%5d for m=%5d, n=%5d, type %2d\n", subnam, info, m, n, imat)

		} else if c3 == "con" {
			fmt.Printf(" *** Error code from %s =%5d for norm='%c', n=%5d, type %2d\n", subnam, info, opts[0], m, imat)

		} else if c3 == "ls" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> trans = '%c', m=%5d, n=%5d, nrhs=%4d, nb=%4d, type %2d\n", subnam, info, opts[0], m, n, kl, n5, imat)

		} else if c3 == "lsx" || c3 == "lss" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> m=%5d, n=%5d, nrhs=%4d, nb=%4d, type %2d\n", subnam, info, m, n, kl, n5, imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> trans = '%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], m, n5, imat)
		}

	} else if p2 == "gb" {
		//        xGB:  General band matrices
		if c3 == "trf" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> m= %5d, n=%5d, kl=%5d, ku=%5d, nb=%4d, type %2d\n", subnam, info, infoe, m, n, kl, ku, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> m= %5d, n=%5d, kl=%5d, ku=%5d, nb=%4d, type %2d\n", subnam, info, m, n, kl, ku, n5, imat)
			}
			if info != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if c3 == "sv" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> n=%5d, kl=%5d, ku=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, n, kl, ku, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> n=%5d, kl=%5d, ku=%5d, nrhs=%4d, type %2d\n", subnam, info, n, kl, ku, n5, imat)
			}

		} else if c3 == "svx" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> fact='%c', trans='%c', n=%5d, kl=%5d, ku=%5d, nrhs=%4d, type %1d\n", subnam, info, infoe, opts[0], opts[1], n, kl, ku, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> fact='%c', trans='%c', n=%5d, kl=%5d, ku=%5d, nrhs=%4d, type %1d\n", subnam, info, opts[0], opts[1], n, kl, ku, n5, imat)
			}

		} else if string(subnam[1:6]) == "latms" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> m= %5d, n=%5d, kl=%5d, ku=%5d, type %2d\n", subnam, info, m, n, kl, ku, imat)

		} else if c3 == "con" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> norm ='%c', n=%5d, kl=%5d, ku=%5d, type %2d\n", subnam, info, opts[0], m, kl, ku, imat)

		} else {
			fmt.Printf(" *** Error code from %s=%5d\n ==> trans='%c', n=%5d, kl=%5d, ku=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], m, kl, ku, n5, imat)
		}

	} else if p2 == "gt" {
		//        xGT:  General tridiagonal matrices
		if c3 == "trf" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d for n=%5d, type %2d\n", subnam, info, infoe, n, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for n=%5d, type %2d\n", subnam, info, n, imat)
			}
			if info != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if c3 == "sv" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for n=%5d, nrhs=%4d, type %2d\n", subnam, info, n, n5, imat)
			}

		} else if c3 == "svx" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> fact='%c', trans='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, opts[0], opts[1], n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> fact='%c', trans='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], opts[1], n, n5, imat)
			}

		} else if c3 == "con" {
			fmt.Printf(" *** Error code from %s =%5d for norm='%c', n=%5d, type %2d\n", subnam, info, opts[0], m, imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> trans = '%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], m, n5, imat)
		}

	} else if p2 == "po" {
		//        xPO:  Symmetric or Hermitian positive definite matrices
		uplo = mat.UploByte(opts[0])
		if c3 == "trf" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, nb=%4d, type %2d\n", subnam, info, infoe, uplo, m, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nb=%4d, type %2d\n", subnam, info, uplo, m, n5, imat)
			}
			if info != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if c3 == "sv" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, uplo, n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, n, n5, imat)
			}

		} else if c3 == "svx" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> fact='%c', uplo='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, opts[0], opts[1], n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> fact='%c', uplo='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], opts[1], n, n5, imat)
			}

		} else if c3 == "tri" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nb=%4d, type %2d\n", subnam, info, uplo, m, n5, imat)

		} else if string(subnam[1:6]) == "latms" || c3 == "con" {
			fmt.Printf(" *** Error code from %s =%5d for uplo=%s, n=%5d, type %2d\n", subnam, info, uplo, m, imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, m, n5, imat)
		}

	} else if p2 == "ps" {
		//        xPS:  Symmetric or Hermitian positive semi-definite matrices
		uplo = mat.UploByte(opts[0])
		if c3 == "trf" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, nb=%4d, type %2d\n", subnam, info, infoe, uplo, m, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nb=%4d, type %2d\n", subnam, info, uplo, m, n5, imat)
			}
			if info != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if c3 == "sv" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, uplo, n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, n, n5, imat)
			}

		} else if c3 == "svx" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> fact='%c', uplo='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, opts[0], opts[1], n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> fact='%c', uplo='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], opts[1], n, n5, imat)
			}

		} else if c3 == "tri" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nb=%4d, type %2d\n", subnam, info, uplo, m, n5, imat)

		} else if string(subnam[1:6]) == "latmt" || c3 == "con" {
			fmt.Printf(" *** Error code from %s =%5d for uplo=%s, n=%5d, type %2d\n", subnam, info, uplo, m, imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, m, n5, imat)
		}

	} else if p2 == "sy" || p2 == "sr" || p2 == "sk" || p2 == "he" || p2 == "hr" || p2 == "hk" || p2 == "ha" {
		//        xSY: symmetric indefinite matrices
		//             with partial (Bunch-Kaufman) pivoting;
		//        xSR: symmetric indefinite matrices
		//             with rook (bounded Bunch-Kaufman) pivoting;
		//        xSK: symmetric indefinite matrices
		//             with rook (bounded Bunch-Kaufman) pivoting,
		//             new storage format;
		//        xHE: Hermitian indefinite matrices
		//             with partial (Bunch-Kaufman) pivoting.
		//        xHR: Hermitian indefinite matrices
		//             with rook (bounded Bunch-Kaufman) pivoting;
		//        xHK: Hermitian indefinite matrices
		//             with rook (bounded Bunch-Kaufman) pivoting,
		//             new storage format;
		//        xHA: Hermitian matrices
		//             Aasen Algorithm
		uplo = mat.UploByte(opts[0])
		if c3 == "trf" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, nb=%4d, type %2d\n", subnam, info, infoe, uplo, m, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nb=%4d, type %2d\n", subnam, info, uplo, m, n5, imat)
			}
			if info != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if c3 == "sv" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, uplo, n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, n, n5, imat)
			}

		} else if c3 == "svx" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> fact='%c', uplo='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, opts[0], opts[1], n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d  want %5d\n ==> fact='%c', uplo='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, opts[0], opts[1], n, n5, imat)
			}

		} else if string(subnam[1:6]) == "latms" || c3 == "tri" || c3 == "con" {
			fmt.Printf(" *** Error code from %s =%5d for uplo=%s, n=%5d, type %2d\n", subnam, info, uplo, m, imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, m, n5, imat)
		}

	} else if p2 == "pp" || p2 == "sp" || p2 == "hp" {
		//        xPP, xHP, or xSP:  Symmetric or Hermitian packed matrices
		uplo = mat.UploByte(opts[0])
		if c3 == "trf" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, type %2d\n", subnam, info, infoe, uplo, m, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for uplo=%s, n=%5d, type %2d\n", subnam, info, uplo, m, imat)
			}
			if info != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if c3 == "sv" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, uplo, n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, n, n5, imat)
			}

		} else if c3 == "svx" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> fact='%c', uplo='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, opts[0], opts[1], n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> fact='%c', uplo='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], opts[1], n, n5, imat)
			}

		} else if string(subnam[1:6]) == "latms" || c3 == "tri" || c3 == "cpn" {
			fmt.Printf(" *** Error code from %s =%5d for uplo=%s, n=%5d, type %2d\n", subnam, info, uplo, m, imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, m, n5, imat)
		}

	} else if p2 == "pb" {
		//        xPB:  Symmetric (Hermitian) positive definite band matrix
		uplo = mat.UploByte(opts[0])
		if c3 == "trf" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, kd =%5d, nb=%4d, type %2d\n", subnam, info, infoe, uplo, m, kl, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, kd =%5d, nb=%4d, type %2d\n", subnam, info, uplo, m, kl, n5, imat)
			}
			if info != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if c3 == "sv" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> uplo=%s, n=%5d, kd =%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, uplo, n, kl, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s=%5d\n ==> uplo=%s, n=%5d, kd =%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, n, kl, n5, imat)
			}

		} else if c3 == "svx" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> fact='%c', uplo='%c', n=%5d, kd=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, opts[0], opts[1], n, kl, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> fact='%c', uplo='%c', n=%5d, kd=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], opts[1], n, kl, n5, imat)
			}

		} else if string(subnam[1:6]) == "latms" || c3 == "con" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo=%s, n=%5d, kd =%5d, type %2d\n", subnam, info, uplo, m, kl, imat)

		} else {
			fmt.Printf(" *** Error code from %s=%5d\n ==> uplo=%s, n=%5d, kd =%5d, nrhs=%4d, type %2d\n", subnam, info, uplo, m, kl, n5, imat)
		}

	} else if p2 == "pt" {
		//        xPT:  Positive definite tridiagonal matrices
		if c3 == "trf" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d for n=%5d, type %2d\n", subnam, info, infoe, n, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for n=%5d, type %2d\n", subnam, info, n, imat)
			}
			if info != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if c3 == "sv" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for n=%5d, nrhs=%4d, type %2d\n", subnam, info, n, n5, imat)
			}

		} else if c3 == "svx" {
			if info != infoe && infoe != 0 {
				fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> fact='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, infoe, opts[0], n, n5, imat)
			} else {
				fmt.Printf(" *** Error code from %s=%5d, fact='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], n, n5, imat)
			}

		} else if c3 == "con" {
			if subnam[0] == 'S' || subnam[0] == 'D' {
				fmt.Printf(" *** Error code from %s =%5d for n=%5d, type %2d\n", subnam, info, m, imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for norm='%c', n=%5d, type %2d\n", subnam, info, opts[0], m, imat)
			}

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> trans = '%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], m, n5, imat)
		}

	} else if p2 == "tr" {
		//        xTR:  Triangular matrix
		if c3 == "tri" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo='%c', diag='%c', n=%5d, nb=%4d, type %2d\n", subnam, info, opts[0], opts[1], m, n5, imat)
		} else if c3 == "cono" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> norm='%c', uplo='%c', diag='%c', n=%5d, type %2d\n", subnam, info, opts[0], opts[1], opts[2], m, imat)
		} else if string(subnam[1:5]) == "latrs" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo='%c', trans='%c', diag='%c', normin='%c', n=%5d, type %2d\n", subnam, info, opts[0], opts[1], opts[2], opts[3], m, imat)
		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo='%c', trans='%c', diag='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], opts[1], opts[2], m, n5, imat)
		}

	} else if p2 == "tp" {
		//        xTP:  Triangular packed matrix
		if c3 == "tri" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo='%c', diag='%c', n=%5d, type %2d\n", subnam, info, opts[0], opts[1], m, imat)
		} else if c3 == "con" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> norm='%c', uplo='%c', diag='%c', n=%5d, type %2d\n", subnam, info, opts[0], opts[1], opts[2], m, imat)
		} else if string(subnam[1:6]) == "latps" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo='%c', trans='%c', diag='%c', normin='%c', n=%5d, type %2d\n", subnam, info, opts[0], opts[1], opts[2], opts[3], m, imat)
		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo='%c', trans='%c', diag='%c', n=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], opts[1], opts[2], m, n5, imat)
		}

	} else if p2 == "tb" {
		//        xTB:  Triangular band matrix
		if c3 == "con" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> norm='%c', uplo='%c', diag='%c', n=%5d, kd=%5d, type %2d\n", subnam, info, opts[0], opts[1], opts[2], m, kl, imat)
		} else if string(subnam[1:6]) == "latbs" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo='%c', trans='%c', diag='%c', normin='%c', n=%5d, kd=%5d, type %2d\n", subnam, info, opts[0], opts[1], opts[2], opts[3], m, kl, imat)
		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> uplo='%c', trans='%c', diag='%c', n=%5d, kd=%5d, nrhs=%4d, type %2d\n", subnam, info, opts[0], opts[1], opts[2], m, kl, n5, imat)
		}

	} else if p2 == "qr" {
		//        xQR:  QR factorization
		if c3 == "qrs" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> m=%5d, n=%5d, nrhs=%4d, nb=%4d, type %2d\n", subnam, info, m, n, kl, n5, imat)
		} else if string(subnam[1:6]) == "latms" {
			fmt.Printf(" *** Error code from %s =%5d for m=%5d, n=%5d, type %2d\n", subnam, info, m, n, imat)
		}

	} else if p2 == "lq" {
		//        xLQ:  LQ factorization
		if c3 == "lqs" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> m=%5d, n=%5d, nrhs=%4d, nb=%4d, type %2d\n", subnam, info, m, n, kl, n5, imat)
		} else if string(subnam[1:5]) == "latms" {
			fmt.Printf(" *** Error code from %s =%5d for m=%5d, n=%5d, type %2d\n", subnam, info, m, n, imat)
		}

	} else if p2 == "ql" {
		//        xQL:  QL factorization
		if c3 == "qls" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> m=%5d, n=%5d, nrhs=%4d, nb=%4d, type %2d\n", subnam, info, m, n, kl, n5, imat)
		} else if string(subnam[1:6]) == "latms" {
			fmt.Printf(" *** Error code from %s =%5d for m=%5d, n=%5d, type %2d\n", subnam, info, m, n, imat)
		}

	} else if p2 == "rq" {
		//        xRQ:  RQ factorization
		if c3 == "rqs" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> m=%5d, n=%5d, nrhs=%4d, nb=%4d, type %2d\n", subnam, info, m, n, kl, n5, imat)
		} else if string(subnam[1:6]) == "latms" {
			fmt.Printf(" *** Error code from %s =%5d for m=%5d, n=%5d, type %2d\n", subnam, info, m, n, imat)
		}

	} else if p2 == "lu" {
		if info != infoe && infoe != 0 {
			fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> m=%5d, n=%5d, nb=%4d, type %2d\n", subnam, info, infoe, m, n, n5, imat)
		} else {
			fmt.Printf(" *** Error code from %s=%5d for m=%5d, n=%5d, nb=%4d, type %2d\n", subnam, info, m, n, n5, imat)
		}

	} else if p2 == "ch" {
		if info != infoe && infoe != 0 {
			fmt.Printf(" *** %s returned with info=%5d instead of %2d\n ==> n=%5d, nb=%4d, type %2d\n", subnam, info, infoe, m, n5, imat)
		} else {
			fmt.Printf(" *** Error code from %s=%5d for n=%5d, nb=%4d, type %2d\n", subnam, info, m, n5, imat)
		}

	} else {
		//        Print a generic message if the path is unknown.
		fmt.Printf(" *** Error code from %s =%5d\n", subnam, info)
	}

	return
}
