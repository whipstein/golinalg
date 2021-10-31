package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dckglm tests DGGGLM - subroutine for solving generalized linear
//                       model problem.
func dckglm(nn int, mval, pval, nval []int, nmats int, iseed []int, thresh float64, nmax int, a, af, b, bf, x, work, rwork *mat.Vector, nout int, t *testing.T) (nfail, nrun int, err error) {
	var firstt bool
	var dista, distb, _type byte
	var anorm, bnorm, cndnma, cndnmb, resid float64
	var i, iinfo, ik, imat, kla, klb, kua, kub, lda, ldb, lwork, m, modea, modeb, n, ntypes, p int
	dotype := make([]bool, 8)

	ntypes = 8

	//     Initialize constants.
	path := "Glm"
	nrun = 0
	nfail = 0
	firstt = true
	alareq(nmats, &dotype)
	lda = nmax
	ldb = nmax
	lwork = nmax * nmax

	//     Check for valid input values.
	for ik = 1; ik <= nn; ik++ {
		m = mval[ik-1]
		p = pval[ik-1]
		n = nval[ik-1]
		if m > n || n > m+p {
			if firstt {
				fmt.Printf("\n")
				firstt = false
			}
			fmt.Printf(" *** Invalid input  for GLM:  M = %6d, P = %6d, N = %6d;\n     must satisfy M <= N <= M+P  (this set of values will be skipped)\n", m, p, n)
		}
	}
	firstt = true

	//     Do for each value of M in MVAL.
	for ik = 1; ik <= nn; ik++ {
		m = mval[ik-1]
		p = pval[ik-1]
		n = nval[ik-1]
		if m > n || n > m+p {
			goto label40
		}

		for imat = 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label30
			}

			//           Set up parameters with DLATB9 and generate test
			//           matrices A and B with DLATMS.
			_type, kla, kua, klb, kub, anorm, bnorm, modea, modeb, cndnma, cndnmb, dista, distb = dlatb9(path, imat, m, p, n)

			if iinfo, err = matgen.Dlatms(n, m, dista, &iseed, _type, rwork, modea, cndnma, anorm, kla, kua, 'N', a.Matrix(lda, opts), work); iinfo != 0 {
				t.Fail()
				fmt.Printf(" DLATMS in DCKGLM INFO = %5d\n", iinfo)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label30
			}

			if iinfo, err = matgen.Dlatms(n, p, distb, &iseed, _type, rwork, modeb, cndnmb, bnorm, klb, kub, 'N', b.Matrix(ldb, opts), work); iinfo != 0 {
				t.Fail()
				fmt.Printf(" DLATMS in DCKGLM INFO = %5d\n", iinfo)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label30
			}

			//           Generate random left hand side vector of GLM
			for i = 1; i <= n; i++ {
				x.Set(i-1, matgen.Dlarnd(2, &iseed))
			}

			resid = dglmts(n, m, p, a.Matrix(lda, opts), af.Matrix(lda, opts), b.Matrix(ldb, opts), bf.Matrix(ldb, opts), x, x.Off(nmax), x.Off(2*nmax), x.Off(3*nmax), work, lwork, rwork)

			//           Print information about the tests that did not
			//           pass the threshold.
			if resid >= thresh {
				t.Fail()
				if nfail == 0 && firstt {
					firstt = false
					alahdg(path)
				}
				fmt.Printf(" N=%4d M=%4d, P=%4d, _type %2d, test %2d, ratio=%13.6f\n", n, m, p, imat, 1, resid)
				nfail++
			}
			nrun++

		label30:
		}
	label40:
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, 0)

	return
}
