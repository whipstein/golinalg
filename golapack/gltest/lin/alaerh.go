package lin

import "fmt"

// Alaerh is an error handler for the LAPACK routines.  It prints the
// header if this is the first error message and prints the error code
// and form of recovery, if any.  The character evaluations in this
// routine may make it slow, but it should not be called once the LAPACK
// routines are fully debugged.
func Alaerh(path, subnam []byte, info, infoe *int, opts []byte, m, n, kl, ku, n5, imat, nfail, nerrs *int) {
	var uplo byte

	if (*info) == 0 {
		return
	}
	p2 := path[1:3]
	c3 := []byte(subnam[3:6])

	//     Print the header if this is the first error message.
	if (*nfail) == 0 && (*nerrs) == 0 {
		if string(c3) == "SV " || string(c3) == "SVX" {
			Aladhd(path)
		} else {
			Alahd(path)
		}
	}
	(*nerrs) = (*nerrs) + 1
	//
	//     Print the message detailing the error and form of recovery,
	//     if any.
	//
	if string(p2) == "GE" {
		//
		//        xGE:  General matrices
		//
		if string(c3) == "TRF" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> M =%5d, N =%5d, NB =%4d, type %2d\n", subnam[1:], *info, *infoe, *m, *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s=%5d for M=%5d, N=%5d, NB=%4d, type %2d\n", subnam[1:], *info, *m, *n, *n5, *imat)
			}
			if (*info) != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if string(c3) == "SV " {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *n, *n5, *imat)
			}

		} else if string(c3) == "SVX" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> FACT='%c', TRANS='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, opts[0], opts[1], *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> FACT='%c', TRANS='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], *n, *n5, *imat)
			}

		} else if string(c3) == "TRI" {
			fmt.Printf(" *** Error code from %s=%5d for N=%5d, NB=%4d, type %2d\n", subnam[1:], *info, *n, *n5, *imat)

		} else if string(subnam[1:6]) == "LATMS" {
			fmt.Printf(" *** Error code from %s =%5d for M =%5d, N =%5d, type %2d\n", subnam[1:], *info, *m, *n, *imat)

		} else if string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d for NORM = '%c', N =%5d, type %2d\n", subnam[1:], *info, opts[0], *m, *imat)

		} else if string(c3) == "LS " {
			fmt.Printf(" *** Error code from %s =%5d\n ==> TRANS = '%c', M =%5d, N =%5d, NRHS =%4d, NB =%4d, type %2d\n", subnam[1:], *info, opts[0], *m, *n, *kl, *n5, *imat)

		} else if string(c3) == "LSX" || string(c3) == "LSS" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> M =%5d, N =%5d, NRHS =%4d, NB =%4d, type %2d\n", subnam[1:], *info, *m, *n, *kl, *n5, *imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> TRANS = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], *m, *n5, *imat)
		}

	} else if string(p2) == "GB" {
		//        xGB:  General band matrices
		if string(c3) == "TRF" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> M = %5d, N =%5d, KL =%5d, KU =%5d, NB =%4d, type %2d\n", subnam[1:], *info, *infoe, *m, *n, *kl, *ku, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> M = %5d, N =%5d, KL =%5d, KU =%5d, NB =%4d, type %2d\n", subnam[1:], *info, *m, *n, *kl, *ku, *n5, *imat)
			}
			if (*info) != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if string(c3) == "SV " {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> N =%5d, KL =%5d, KU =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, *n, *kl, *ku, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> N =%5d, KL =%5d, KU =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *n, *kl, *ku, *n5, *imat)
			}

		} else if string(c3) == "SVX" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> FACT='%c', TRANS='%c', N=%5d, KL=%5d, KU=%5d, NRHS=%4d, type %1d\n", subnam[1:], *info, *infoe, opts[0], opts[1], *n, *kl, *ku, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> FACT='%c', TRANS='%c', N=%5d, KL=%5d, KU=%5d, NRHS=%4d, type %1d\n", subnam[1:], *info, opts[0], opts[1], *n, *kl, *ku, *n5, *imat)
			}

		} else if string(subnam[1:6]) == "LATMS" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> M = %5d, N =%5d, KL =%5d, KU =%5d, type %2d\n", subnam[1:], *info, *m, *n, *kl, *ku, *imat)

		} else if string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> NORM ='%c', N =%5d, KL =%5d, KU =%5d, type %2d\n", subnam[1:], *info, opts[0], *m, *kl, *ku, *imat)

		} else {
			fmt.Printf(" *** Error code from %s=%5d\n ==> TRANS='%c', N =%5d, KL =%5d, KU =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], *m, *kl, *ku, *n5, *imat)
		}

	} else if string(p2) == "GT" {
		//        xGT:  General tridiagonal matrices
		if string(c3) == "TRF" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d for N=%5d, type %2d\n", subnam[1:], *info, *infoe, *n, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for N =%5d, type %2d\n", subnam[1:], *info, *n, *imat)
			}
			if (*info) != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if string(c3) == "SV " {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *n, *n5, *imat)
			}

		} else if string(c3) == "SVX" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> FACT='%c', TRANS='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, opts[0], opts[1], *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> FACT='%c', TRANS='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], *n, *n5, *imat)
			}

		} else if string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d for NORM = '%c', N =%5d, type %2d\n", subnam[1:], *info, opts[0], *m, *imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> TRANS = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], *m, *n5, *imat)
		}

	} else if string(p2) == "PO" {
		//        xPO:  Symmetric or Hermitian positive definite matrices
		uplo = opts[0]
		if string(c3) == "TRF" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO = '%c', N =%5d, NB =%4d, type %2d\n", subnam[1:], *info, *infoe, uplo, *m, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NB =%4d, type %2d\n", subnam[1:], *info, uplo, *m, *n5, *imat)
			}
			if (*info) != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if string(c3) == "SV " {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, uplo, *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, uplo, *n, *n5, *imat)
			}

		} else if string(c3) == "SVX" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> FACT='%c', UPLO='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, opts[0], opts[1], *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> FACT='%c', UPLO='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], *n, *n5, *imat)
			}

		} else if string(c3) == "TRI" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NB =%4d, type %2d\n", subnam[1:], *info, uplo, *m, *n5, *imat)

		} else if string(subnam[1:6]) == "LATMS" || string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d for UPLO = '%c', N =%5d, type %2d\n", subnam[1:], *info, uplo, *m, *imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, uplo, *m, *n5, *imat)
		}

	} else if string(p2) == "PS" {
		//        xPS:  Symmetric or Hermitian positive semi-definite matrices
		uplo = opts[0]
		if string(c3) == "TRF" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO = '%c', N =%5d, NB =%4d, type %2d\n", subnam, *info, *infoe, uplo, *m, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NB =%4d, type %2d\n", subnam, *info, uplo, *m, *n5, *imat)
			}
			if (*info) != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if string(c3) == "SV " {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam, *info, *infoe, uplo, *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam, *info, uplo, *n, *n5, *imat)
			}

		} else if string(c3) == "SVX" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> FACT='%c', UPLO='%c', N =%5d, NRHS =%4d, type %2d\n", subnam, *info, *infoe, opts[0], opts[1], *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> FACT='%c', UPLO='%c', N =%5d, NRHS =%4d, type %2d\n", subnam, *info, opts[0], opts[1], *n, *n5, *imat)
			}

		} else if string(c3) == "TRI" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NB =%4d, type %2d\n", subnam, *info, uplo, *m, *n5, *imat)

		} else if string(subnam[1:6]) == "LATMT" || string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d for UPLO = '%c', N =%5d, type %2d\n", subnam, *info, uplo, *m, *imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam, *info, uplo, *m, *n5, *imat)
		}

	} else if string(p2) == "SY" || string(p2) == "SR" || string(p2) == "SK" || string(p2) == "HE" || string(p2) == "HR" || string(p2) == "HK" || string(p2) == "HA" {
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
		uplo = opts[0]
		if string(c3) == "TRF" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO = '%c', N =%5d, NB =%4d, type %2d\n", subnam[1:], *info, *infoe, uplo, *m, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NB =%4d, type %2d\n", subnam[1:], *info, uplo, *m, *n5, *imat)
			}
			if (*info) != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if string(c3) == "SV" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, uplo, *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, uplo, *n, *n5, *imat)
			}

		} else if string(c3) == "SVX" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> FACT='%c', UPLO='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, opts[0], opts[1], *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> FACT='%c', UPLO='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], *n, *n5, *imat)
			}

		} else if string(subnam[1:6]) == "LATMS" || string(c3) == "TRI" || string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d for UPLO = '%c', N =%5d, type %2d\n", subnam[1:], *info, uplo, *m, *imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, uplo, *m, *n5, *imat)
		}

	} else if string(p2) == "PP" || string(p2) == "SP" || string(p2) == "HP" {
		//        xPP, xHP, or xSP:  Symmetric or Hermitian packed matrices
		uplo = opts[0]
		if string(c3) == "TRF" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO = '%c', N =%5d, type %2d\n", subnam[1:], *info, *infoe, uplo, *m, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for UPLO = '%c', N =%5d, type %2d\n", subnam[1:], *info, uplo, *m, *imat)
			}
			if (*info) != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if string(c3) == "SV " {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, uplo, *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, uplo, *n, *n5, *imat)
			}

		} else if string(c3) == "SVX" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> FACT='%c', UPLO='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, opts[0], opts[1], *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> FACT='%c', UPLO='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], *n, *n5, *imat)
			}

		} else if string(subnam[1:6]) == "LATMS" || string(c3) == "TRI" || string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d for UPLO = '%c', N =%5d, type %2d\n", subnam[1:], *info, uplo, *m, *imat)

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, uplo, *m, *n5, *imat)
		}

	} else if string(p2) == "PB" {
		//        xPB:  Symmetric (Hermitian) positive definite band matrix
		uplo = opts[0]
		if string(c3) == "TRF" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO = '%c', N =%5d, KD =%5d, NB =%4d, type %2d\n", subnam[1:], *info, *infoe, uplo, *m, *kl, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, KD =%5d, NB =%4d, type %2d\n", subnam[1:], *info, uplo, *m, *kl, *n5, *imat)
			}
			if (*info) != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if string(c3) == "SV " {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> UPLO='%c', N =%5d, KD =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, uplo, *n, *kl, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s=%5d\n ==> UPLO = '%c', N =%5d, KD =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, uplo, *n, *kl, *n5, *imat)
			}

		} else if string(c3) == "SVX" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> FACT='%c', UPLO='%c', N=%5d, KD=%5d, NRHS=%4d, type %2d\n", subnam[1:], *info, *infoe, opts[0], opts[1], *n, *kl, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d\n ==> FACT='%c', UPLO='%c', N=%5d, KD=%5d, NRHS=%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], *n, *kl, *n5, *imat)
			}

		} else if string(subnam[1:6]) == "LATMS" || string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO = '%c', N =%5d, KD =%5d, type %2d\n", subnam[1:], *info, uplo, *m, *kl, *imat)

		} else {
			fmt.Printf(" *** Error code from %s=%5d\n ==> UPLO = '%c', N =%5d, KD =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, uplo, *m, *kl, *n5, *imat)
		}

	} else if string(p2) == "PT" {
		//        xPT:  Positive definite tridiagonal matrices
		if string(c3) == "TRF" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d for N=%5d, type %2d\n", subnam[1:], *info, *infoe, *n, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for N =%5d, type %2d\n", subnam[1:], *info, *n, *imat)
			}
			if (*info) != 0 {
				fmt.Printf(" ==> Doing only the condition estimate for this case\n")
			}

		} else if string(c3) == "SV " {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *n, *n5, *imat)
			}

		} else if string(c3) == "SVX" {
			if (*info) != (*infoe) && (*infoe) != 0 {
				fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> FACT='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, *infoe, opts[0], *n, *n5, *imat)
			} else {
				fmt.Printf(" *** Error code from %s=%5d, FACT='%c', N=%5d, NRHS=%4d, type %2d\n", subnam[1:], *info, opts[0], *n, *n5, *imat)
			}

		} else if string(c3) == "CON" {
			if subnam[0] == 'S' || subnam[0] == 'D' {
				fmt.Printf(" *** Error code from %s =%5d for N =%5d, type %2d\n", subnam[1:], *info, *m, *imat)
			} else {
				fmt.Printf(" *** Error code from %s =%5d for NORM = '%c', N =%5d, type %2d\n", subnam[1:], *info, opts[0], *m, *imat)
			}

		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> TRANS = '%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], *m, *n5, *imat)
		}

	} else if string(p2) == "TR" {
		//        xTR:  Triangular matrix
		if string(c3) == "TRI" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO='%c', DIAG ='%c', N =%5d, NB =%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], *m, *n5, *imat)
		} else if string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> NORM='%c', UPLO ='%c', DIAG='%c', N =%5d, type %2d\n", subnam[1:], *info, opts[0], opts[1], opts[2], *m, *imat)
		} else if string(subnam[1:5]) == "LATRS" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO='%c', TRANS='%c', DIAG='%c', NORMIN='%c', N =%5d, type %2d\n", subnam[1:], *info, opts[0], opts[1], opts[2], opts[3], *m, *imat)
		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO='%c', TRANS='%c', DIAG='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], opts[2], *m, *n5, *imat)
		}

	} else if string(p2) == "TP" {
		//        xTP:  Triangular packed matrix
		if string(c3) == "TRI" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO='%c', DIAG ='%c', N =%5d, type %2d\n", subnam[1:], *info, opts[0], opts[1], *m, *imat)
		} else if string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> NORM='%c', UPLO ='%c', DIAG='%c', N =%5d, type %2d\n", subnam[1:], *info, opts[0], opts[1], opts[2], *m, *imat)
		} else if string(subnam[1:6]) == "LATPS" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO='%c', TRANS='%c', DIAG='%c', NORMIN='%c', N =%5d, type %2d\n", subnam[1:], *info, opts[0], opts[1], opts[2], opts[3], *m, *imat)
		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO='%c', TRANS='%c', DIAG='%c', N =%5d, NRHS =%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], opts[2], *m, *n5, *imat)
		}

	} else if string(p2) == "TB" {
		//        xTB:  Triangular band matrix
		if string(c3) == "CON" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> NORM='%c', UPLO ='%c', DIAG='%c', N=%5d, KD=%5d, type %2d\n", subnam[1:], *info, opts[0], opts[1], opts[2], *m, *kl, *imat)
		} else if string(subnam[1:6]) == "LATBS" {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO='%c', TRANS='%c', DIAG='%c', NORMIN='%c', N=%5d, KD=%5d, type %2d\n", subnam[1:], *info, opts[0], opts[1], opts[2], opts[3], *m, *kl, *imat)
		} else {
			fmt.Printf(" *** Error code from %s =%5d\n ==> UPLO='%c', TRANS='%c', DIAG='%c', N=%5d, KD=%5d, NRHS=%4d, type %2d\n", subnam[1:], *info, opts[0], opts[1], opts[2], *m, *kl, *n5, *imat)
		}

	} else if string(p2) == "QR" {
		//        xQR:  QR factorization
		if string(c3) == "QRS" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> M =%5d, N =%5d, NRHS =%4d, NB =%4d, type %2d\n", subnam[1:], *info, *m, *n, *kl, *n5, *imat)
		} else if string(subnam[1:6]) == "LATMS" {
			fmt.Printf(" *** Error code from %s =%5d for M =%5d, N =%5d, type %2d\n", subnam[1:], *info, *m, *n, *imat)
		}

	} else if string(p2) == "LQ" {
		//        xLQ:  LQ factorization
		if string(c3) == "LQS" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> M =%5d, N =%5d, NRHS =%4d, NB =%4d, type %2d\n", subnam[1:], *info, *m, *n, *kl, *n5, *imat)
		} else if string(subnam[1:5]) == "LATMS" {
			fmt.Printf(" *** Error code from %s =%5d for M =%5d, N =%5d, type %2d\n", subnam[1:], *info, *m, *n, *imat)
		}

	} else if string(p2) == "QL" {
		//        xQL:  QL factorization
		if string(c3) == "QLS" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> M =%5d, N =%5d, NRHS =%4d, NB =%4d, type %2d\n", subnam[1:], *info, *m, *n, *kl, *n5, *imat)
		} else if string(subnam[1:6]) == "LATMS" {
			fmt.Printf(" *** Error code from %s =%5d for M =%5d, N =%5d, type %2d\n", subnam[1:], *info, *m, *n, *imat)
		}

	} else if string(p2) == "RQ" {
		//        xRQ:  RQ factorization
		if string(c3) == "RQS" {
			fmt.Printf(" *** Error code from %s=%5d\n ==> M =%5d, N =%5d, NRHS =%4d, NB =%4d, type %2d\n", subnam[1:], *info, *m, *n, *kl, *n5, *imat)
		} else if string(subnam[1:6]) == "LATMS" {
			fmt.Printf(" *** Error code from %s =%5d for M =%5d, N =%5d, type %2d\n", subnam[1:], *info, *m, *n, *imat)
		}

	} else if string(p2) == "LU" {
		if (*info) != (*infoe) && (*infoe) != 0 {
			fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> M =%5d, N =%5d, NB =%4d, type %2d\n", subnam[1:], *info, *infoe, *m, *n, *n5, *imat)
		} else {
			fmt.Printf(" *** Error code from %s=%5d for M=%5d, N=%5d, NB=%4d, type %2d\n", subnam[1:], *info, *m, *n, *n5, *imat)
		}

	} else if string(p2) == "CH" {
		if (*info) != (*infoe) && (*infoe) != 0 {
			fmt.Printf(" *** %s returned with INFO =%5d instead of %2d\n ==> N =%5d, NB =%4d, type %2d\n", subnam[1:], *info, *infoe, *m, *n5, *imat)
		} else {
			fmt.Printf(" *** Error code from %s=%5d for N=%5d, NB=%4d, type %2d\n", subnam[1:], *info, *m, *n5, *imat)
		}

	} else {
		//        Print a generic message if the path is unknown.
		fmt.Printf(" *** Error code from %s =%5d\n", subnam[1:], *info)
	}
}
