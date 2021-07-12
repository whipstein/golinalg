package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zckglm tests ZGGGLM - subroutine for solving generalized linear
//                       model problem.
func Zckglm(nn *int, nval, mval, pval *[]int, nmats *int, iseed *[]int, thresh *float64, nmax *int, a, af, b, bf, x, work *mat.CVector, rwork *mat.Vector, nout, info *int, t *testing.T) {
	var firstt bool
	var dista, distb, _type byte
	var anorm, bnorm, cndnma, cndnmb, resid float64
	var i, iinfo, ik, imat, kla, klb, kua, kub, lda, ldb, lwork, m, modea, modeb, n, nfail, nrun, ntypes, p int
	dotype := make([]bool, 8)

	ntypes = 8

	//     Initialize constants.
	path := []byte("GLM")
	(*info) = 0
	nrun = 0
	nfail = 0
	firstt = true
	Alareq(nmats, &dotype)
	lda = (*nmax)
	ldb = (*nmax)
	lwork = (*nmax) * (*nmax)

	//     Check for valid input values.
	for ik = 1; ik <= (*nn); ik++ {
		m = (*mval)[ik-1]
		p = (*pval)[ik-1]
		n = (*nval)[ik-1]
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
	for ik = 1; ik <= (*nn); ik++ {
		m = (*mval)[ik-1]
		p = (*pval)[ik-1]
		n = (*nval)[ik-1]
		if m > n || n > m+p {
			goto label40
		}

		for imat = 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label30
			}

			//           Set up parameters with DLATB9 and generate test
			//           matrices A and B with ZLATMS.
			Dlatb9(path, &imat, &m, &p, &n, &_type, &kla, &kua, &klb, &kub, &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb, &dista, &distb)

			matgen.Zlatms(&n, &m, dista, iseed, _type, rwork, &modea, &cndnma, &anorm, &kla, &kua, 'N', a.CMatrix(lda, opts), &lda, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZLATMS in ZCKGLM INFO = %5d\n", iinfo)
				(*info) = abs(iinfo)
				goto label30
			}

			matgen.Zlatms(&n, &p, distb, iseed, _type, rwork, &modeb, &cndnmb, &bnorm, &klb, &kub, 'N', b.CMatrix(ldb, opts), &ldb, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZLATMS in ZCKGLM INFO = %5d\n", iinfo)
				(*info) = abs(iinfo)
				goto label30
			}

			//           Generate random left hand side vector of GLM
			for i = 1; i <= n; i++ {
				x.Set(i-1, matgen.Zlarnd(func() *int { y := 2; return &y }(), iseed))
			}

			Zglmts(&n, &m, &p, a.CMatrix(lda, opts), af.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), bf.CMatrix(ldb, opts), &ldb, x, x.Off((*nmax)), x.Off(2*(*nmax)), x.Off(3*(*nmax)), work, &lwork, rwork, &resid)

			//           Print information about the tests that did not
			//           pass the threshold.
			if resid >= (*thresh) {
				t.Fail()
				if nfail == 0 && firstt {
					firstt = false
					Alahdg(path)
				}
				fmt.Printf(" N=%4d M=%4d, P=%4d, _type %2d, test %2d, ratio=%13.6f\n", n, m, p, imat, 1, resid)
				nfail = nfail + 1
			}
			nrun = nrun + 1

		label30:
		}
	label40:
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, func() *int { y := 0; return &y }())
}
