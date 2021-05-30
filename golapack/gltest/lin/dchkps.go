package lin

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"math"
	"testing"
)

// Dchkps tests DPSTRF.
func Dchkps(dotype *[]bool, nn *int, nval *[]int, nnb *int, nbval *[]int, nrank *int, rankval *[]int, thresh *float64, tsterr *bool, nmax *int, a, afac, perm *mat.Vector, piv *[]int, work, rwork *mat.Vector, nout *int, t *testing.T) {
	var dist, _type, uplo byte
	var anorm, cndnum, one, result, tol float64
	var comprank, i, imat, in, inb, info, irank, iuplo, izero, kl, ku, lda, mode, n, nb, nerrs, nfail, nimat, nrun, ntypes, rank, rankdiff int

	uplos := make([]byte, 2)
	iseed := make([]int, 4)
	iseedy := make([]int, 4)
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	ntypes = 9

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991
	uplos[0], uplos[1] = 'U', 'L'

	//     Initialize constants and the random number seed.
	path := []byte("DPS")
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if *tsterr {
		Derrps(path, t)
	}
	(*infot) = 0
	Xlaenv(2, 2)

	//     Do for each value of N in NVAL
	for in = 1; in <= (*nn); in++ {
		n = (*nval)[in-1]
		lda = maxint(n, 1)
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		izero = 0
		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !(*dotype)[imat-1] {
				goto label140
			}

			//              Do for each value of RANK in RANKVAL
			for irank = 1; irank <= (*nrank); irank++ {
				//              Only repeat test 3 to 5 for different ranks
				//              Other tests use full rank
				if (imat < 3 || imat > 5) && irank > 1 {
					goto label130
				}

				rank = int(math.Ceil((float64(n) * float64((*rankval)[irank-1])) / 100.))

				//
				//           Do first for UPLO = 'U', then for UPLO = 'L'
				for iuplo = 1; iuplo <= 2; iuplo++ {
					uplo = uplos[iuplo-1]

					//              Set up parameters with DLATB5 and generate a test matrix
					//              with DLATMT.
					Dlatb5(path, &imat, &n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)

					*srnamt = "DLATMT"
					matgen.Dlatmt(&n, &n, dist, &iseed, _type, rwork, &mode, &cndnum, &anorm, &rank, &kl, &ku, uplo, a.Matrix(lda, opts), &lda, work, &info)

					//              Check error code from DLATMT.
					if info != 0 {
						Alaerh(path, []byte("DLATMT"), &info, toPtr(0), []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), toPtr(-1), &imat, &nfail, &nerrs)
						goto label120
					}

					//              Do for each value of NB in NBVAL
					for inb = 1; inb <= (*nnb); inb++ {
						nb = (*nbval)[inb-1]
						Xlaenv(1, nb)

						//                 Compute the pivoted L*L' or U'*U factorization
						//                 of the matrix.
						golapack.Dlacpy(uplo, &n, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda)
						*srnamt = "DPSTRF"

						//                 Use default tolerance
						tol = -one
						golapack.Dpstrf(uplo, &n, afac.Matrix(lda, opts), &lda, piv, &comprank, &tol, work, &info)

						//                 Check error code from DPSTRF.
						if (info < izero) || (info != izero && rank == n) || (info <= izero && rank < n) {
							Alaerh(path, []byte("DPSTRF"), &info, &izero, []byte{uplo}, &n, &n, toPtr(-1), toPtr(-1), &nb, &imat, &nfail, &nerrs)
							goto label110
						}

						//                 Skip the test if INFO is not 0.
						if info != 0 {
							goto label110
						}

						//                 Reconstruct matrix from factors and compute residual.
						//
						//                 PERM holds permuted L*L^T or U^T*U
						Dpst01(uplo, &n, a.Matrix(lda, opts), &lda, afac.Matrix(lda, opts), &lda, perm.Matrix(lda, opts), &lda, piv, rwork, &result, &comprank)

						//                 Print information about the tests that did not pass
						//                 the threshold or where computed rank was not RANK.
						if n == 0 {
							comprank = 0
						}
						rankdiff = rank - comprank
						if result >= (*thresh) {
							if nfail == 0 && nerrs == 0 {
								Alahd(path)
							}
							t.Fail()
							fmt.Printf(" UPLO = '%c', N =%5d, RANK =%3d, Diff =%5d, NB =%4d, _type %2d, Ratio =%12.5f\n", uplo, n, rank, rankdiff, nb, imat, result)
							nfail = nfail + 1
						}
						nrun = nrun + 1
					label110:
					}

				label120:
				}
			label130:
			}
		label140:
		}
	}

	//     Verify number of tests match original.
	tgtRuns := 150
	if nrun != tgtRuns {
		t.Fail()
		fmt.Printf(" Number of tests do not match: got %v, want %v\n", nrun, tgtRuns)
		nerrs++
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, &nerrs)
}
