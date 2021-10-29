package lin

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zchkps tests Zpstrf.
func zchkps(dotype []bool, nn int, nval []int, nnb int, nbval []int, nrank int, rankval []int, thresh float64, tsterr bool, nmax int, a, afac, perm *mat.CVector, piv []int, work *mat.CVector, rwork *mat.Vector, t *testing.T) {
	var dist, _type byte
	var uplo mat.MatUplo
	var anorm, cndnum, one, result, tol float64
	var comprank, i, imat, in, inb, info, irank, izero, kl, ku, lda, mode, n, nb, nerrs, nfail, nimat, nrun, ntypes, rank, rankdiff int
	var err error

	iseed := make([]int, 4)
	iseedy := make([]int, 4)

	one = 1.0
	ntypes = 9
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseedy[0], iseedy[1], iseedy[2], iseedy[3] = 1988, 1989, 1990, 1991

	//     Initialize constants and the random number seed.
	path := "Zps"
	nrun = 0
	nfail = 0
	nerrs = 0
	for i = 1; i <= 4; i++ {
		iseed[i-1] = iseedy[i-1]
	}

	//     Test the error exits
	if tsterr {
		zerrps(path, t)
	}
	(*infot) = 0

	//     Do for each value of N in NVAL
	for in = 1; in <= nn; in++ {
		n = nval[in-1]
		lda = max(n, 1)
		nimat = ntypes
		if n <= 0 {
			nimat = 1
		}

		izero = 0
		for imat = 1; imat <= nimat; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label140
			}

			//              Do for each value of RANK in RANKVAL
			for irank = 1; irank <= nrank; irank++ {
				//              Only repeat test 3 to 5 for different ranks
				//              Other tests use full rank
				if (imat < 3 || imat > 5) && irank > 1 {
					goto label130
				}

				rank = int(math.Ceil((float64(n * rankval[irank-1])) / 100.))

				//           Do first for uplo='U', then for uplo='L'
				for _, uplo = range mat.IterMatUplo(false) {

					//              Set up parameters with ZLATB5 and generate a test matrix
					//              with Zlatmt.
					_type, kl, ku, anorm, mode, cndnum, dist = zlatb5(path, imat, n)

					*srnamt = "Zlatmt"
					if err = matgen.Zlatmt(n, n, dist, &iseed, _type, rwork, mode, cndnum, anorm, rank, kl, ku, uplo.Byte(), a.CMatrix(lda, opts), work); err != nil {
						t.Fail()
						nerrs = alaerh(path, "Zlatmt", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, -1, imat, nfail, nerrs)
						goto label120
					}

					//              Do for each value of NB in NBVAL
					for inb = 1; inb <= nnb; inb++ {
						nb = nbval[inb-1]
						xlaenv(1, nb)

						//                 Compute the pivoted L*L' or U'*U factorization
						//                 of the matrix.
						golapack.Zlacpy(uplo, n, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts))
						*srnamt = "Zpstrf"

						//                 Use default tolerance
						tol = -one
						if comprank, info, err = golapack.Zpstrf(uplo, n, afac.CMatrix(lda, opts), &piv, tol, rwork); err != nil || (info < izero) || (info != izero && rank == n) || (info <= izero && rank < n) {
							nerrs = alaerh(path, "Zpstrf", info, 0, []byte{uplo.Byte()}, n, n, -1, -1, nb, imat, nfail, nerrs)
							goto label110
						}

						//                 Skip the test if INFO is not 0.
						if info != 0 {
							goto label110
						}

						//                 Reconstruct matrix from factors and compute residual.
						//
						//                 PERM holds permuted L*L^T or U^T*U
						*&result = zpst01(uplo, n, a.CMatrix(lda, opts), afac.CMatrix(lda, opts), perm.CMatrix(lda, opts), &piv, rwork, comprank)

						//                 Print information about the tests that did not pass
						//                 the threshold or where computed rank was not RANK.
						if n == 0 {
							comprank = 0
						}
						rankdiff = rank - comprank
						if result >= thresh {
							t.Fail()
							if nfail == 0 && nerrs == 0 {
								alahd(path)
							}
							fmt.Printf(" uplo=%s, n=%5d, rank=%3d, Diff =%5d, nb=%4d, _type %2d, Ratio =%12.5f\n", uplo, n, rank, rankdiff, nb, imat, result)
							nfail++
						}
						nrun++
					label110:
					}

				label120:
				}
			label130:
			}
		label140:
		}
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, nerrs)
}
